package codefreeo

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/chenhg5/cc-connect/core"
)

// codefreeoSession manages multi-turn conversations with the Codefree-O CLI.
// Each Send() launches a new `codefree-o run --format json` process
// with --session for conversation continuity.
type codefreeoSession struct {
	cmd               string
	workDir           string
	model             string
	mode              string
	extraEnv          []string
	events            chan core.Event
	chatID            atomic.Value // stores string — Codefree-O session ID
	ctx               context.Context
	cancel            context.CancelFunc
	wg                sync.WaitGroup
	alive             atomic.Bool
	expectingContinue atomic.Bool // true when compaction_continue received, waiting for next step
}

func newCodefreeoSession(ctx context.Context, cmd, workDir, model, mode, resumeID string, extraEnv []string) (*codefreeoSession, error) {
	sessionCtx, cancel := context.WithCancel(ctx)

	s := &codefreeoSession{
		cmd:      cmd,
		workDir:  workDir,
		model:    model,
		mode:     mode,
		extraEnv: extraEnv,
		events:   make(chan core.Event, 64),
		ctx:      sessionCtx,
		cancel:   cancel,
	}
	s.alive.Store(true)

	if resumeID != "" && resumeID != core.ContinueSession {
		s.chatID.Store(resumeID)
	}

	return s, nil
}

func (s *codefreeoSession) Send(prompt string, images []core.ImageAttachment, files []core.FileAttachment) error {
	if len(files) > 0 {
		filePaths := core.SaveFilesToDisk(s.workDir, files)
		prompt = core.AppendFileRefs(prompt, filePaths)
	}
	prompt, imagePaths, err := s.stageImages(prompt, images)
	if err != nil {
		return err
	}
	if !s.alive.Load() {
		return fmt.Errorf("session is closed")
	}

	chatID := s.CurrentSessionID()
	isResume := chatID != ""

	args := s.buildRunArgs(prompt, imagePaths, chatID)

	slog.Debug("codefreeoSession: launching", "resume", isResume, "args", core.RedactArgs(args))

	cmd := exec.CommandContext(s.ctx, s.cmd, args...)
	cmd.Dir = s.workDir
	env := os.Environ()
	if len(s.extraEnv) > 0 {
		env = core.MergeEnv(env, s.extraEnv)
	}
	cmd.Env = env

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("codefreeoSession: stdout pipe: %w", err)
	}

	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("codefreeoSession: start: %w", err)
	}

	s.wg.Add(1)
	go s.readLoop(cmd, stdout, &stderrBuf)

	return nil
}

func (s *codefreeoSession) stageImages(prompt string, images []core.ImageAttachment) (string, []string, error) {
	if len(images) == 0 {
		return prompt, nil, nil
	}

	imgDir := filepath.Join(s.workDir, ".cc-connect", "images")
	if err := os.MkdirAll(imgDir, 0o755); err != nil {
		return "", nil, fmt.Errorf("codefreeoSession: create image dir: %w", err)
	}

	imagePaths := make([]string, 0, len(images))
	for i, img := range images {
		ext := codefreeoImageExt(img.MimeType)
		fname := fmt.Sprintf("img_%d_%d%s", time.Now().UnixMilli(), i, ext)
		fpath := filepath.Join(imgDir, fname)
		if err := os.WriteFile(fpath, img.Data, 0o644); err != nil {
			return "", nil, fmt.Errorf("codefreeoSession: save image: %w", err)
		}
		imagePaths = append(imagePaths, fpath)
	}

	if prompt == "" {
		prompt = "Please analyze the attached image(s)."
	}

	return prompt, imagePaths, nil
}

func codefreeoImageExt(mimeType string) string {
	switch mimeType {
	case "image/jpeg":
		return ".jpg"
	case "image/gif":
		return ".gif"
	case "image/webp":
		return ".webp"
	default:
		return ".png"
	}
}

func (s *codefreeoSession) buildRunArgs(prompt string, imagePaths []string, chatID string) []string {
	args := []string{"run", "--format", "json"}

	if chatID != "" {
		args = append(args, "--session", chatID)
	}
	if s.model != "" {
		args = append(args, "--model", s.model)
	}
	if s.workDir != "" {
		args = append(args, "--dir", s.workDir)
	}

	// Enable thinking blocks.
	args = append(args, "--thinking")

	for _, imagePath := range imagePaths {
		if imagePath == "" {
			continue
		}
		args = append(args, "--file", imagePath)
	}

	// Use "--" to separate flags from the positional prompt so that
	// --file (yargs [array]) does not greedily consume the prompt text.
	args = append(args, "--", prompt)
	return args
}

func (s *codefreeoSession) readLoop(cmd *exec.Cmd, stdout io.ReadCloser, stderrBuf *bytes.Buffer) {
	defer s.wg.Done()
	defer func() { _ = cmd.Wait() }()

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var raw map[string]any
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			slog.Debug("codefreeoSession: non-JSON line", "line", line)
			continue
		}

		s.handleEvent(raw)
	}

	if err := scanner.Err(); err != nil {
		slog.Error("codefreeoSession: scanner error", "error", err)
		evt := core.Event{Type: core.EventError, Error: fmt.Errorf("read stdout: %w", err)}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
		return
	}

	stderrMsg := stderrBuf.String()
	if stderrMsg != "" {
		slog.Error("codefreeoSession: process error", "stderr", truncate(stderrMsg, 500))
		if strings.Contains(stderrMsg, "Session not found") {
			s.chatID.Store("")
			slog.Warn("codefreeoSession: cleared stale session ID")
		}
		evt := core.Event{Type: core.EventError, Error: fmt.Errorf("%s", stderrMsg)}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
		}
		return
	}

	// Check if we received compaction_continue before readLoop ended.
	// If so, Codefree-O will continue with a new turn - do NOT send EventResult.
	// The subsequent process will send its own EventResult when it finishes.
	if s.expectingContinue.Load() {
		slog.Info("codefreeoSession: readLoop ended after compaction_continue, skipping EventResult", "session_id", s.CurrentSessionID())
		s.expectingContinue.Store(false)
		return
	}

	// Emit EventResult after all steps are done and the process has finished writing.
	sid := s.CurrentSessionID()
	slog.Debug("codefreeoSession: readLoop complete, sending fallback EventResult", "session_id", sid)
	evt := core.Event{Type: core.EventResult, SessionID: sid, Done: true}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
	}
}

// Codefree-O NDJSON event structure:
//
//	{ "type": "text|tool_use|reasoning|step_start|step_finish",
//	  "part": { "type": "text|tool|reasoning|step-start|step-finish", ... } }
func (s *codefreeoSession) handleEvent(raw map[string]any) {
	eventType, _ := raw["type"].(string)

	switch eventType {
	case "text":
		s.handleText(raw)
	case "tool_use":
		s.handleToolUse(raw)
	case "reasoning":
		s.handleReasoning(raw)
	case "step_start":
		s.handleStepStart(raw)
	case "step_finish":
		s.handleStepFinish(raw)
	case "error":
		s.handleError(raw)
	default:
		b, _ := json.Marshal(raw)
		slog.Debug("codefreeoSession: unhandled event", "type", eventType, "raw", string(b))
	}
}

func (s *codefreeoSession) handleText(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	if part == nil {
		return
	}
	text, _ := part["text"].(string)

	// Extract metadata and synthetic flags to identify compaction_continue
	metadata, _ := part["metadata"].(map[string]any)
	synthetic, _ := part["synthetic"].(bool)

	// Check for compaction_continue: this is Codefree-O's auto-continuation signal.
	// When received, we should NOT send EventText to engine, but mark that we expect
	// a continuation (next step_start will start a new turn without EventResult).
	if synthetic && metadata != nil {
		if cc, ok := metadata["compaction_continue"].(bool); ok && cc {
			slog.Info("codefreeoSession: compaction_continue detected, marking expectingContinue", "session_id", s.CurrentSessionID())
			s.expectingContinue.Store(true)
			// Do NOT send EventText - this is internal continuation signal
			return
		}
	}

	if text != "" {
		evt := core.Event{Type: core.EventText, Content: text, Metadata: metadata, Synthetic: synthetic}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *codefreeoSession) handleToolUse(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	if part == nil {
		return
	}

	toolName, _ := part["tool"].(string)

	state, _ := part["state"].(map[string]any)
	status := ""
	if state != nil {
		status, _ = state["status"].(string)
	}

	// Extract tool input summary for display
	input := extractToolInput(state)

	if status == "completed" {
		// Codefree-O bundles call + result in one event; emit both for UI.
		useEvt := core.Event{Type: core.EventToolUse, ToolName: toolName, ToolInput: input}
		select {
		case s.events <- useEvt:
		case <-s.ctx.Done():
			return
		}

		output, _ := state["output"].(string)
		resultEvt := core.Event{Type: core.EventToolResult, ToolName: toolName, Content: truncate(output, 500)}
		select {
		case s.events <- resultEvt:
		case <-s.ctx.Done():
			return
		}
	} else {
		evt := core.Event{Type: core.EventToolUse, ToolName: toolName, ToolInput: input}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
	}
}

func extractToolInput(state map[string]any) string {
	if state == nil {
		return ""
	}
	// Prefer title as a concise description (e.g. "List files in current directory")
	if title, ok := state["title"].(string); ok && title != "" {
		return title
	}
	switch input := state["input"].(type) {
	case string:
		return input
	case map[string]any:
		// Use "description" or "command" fields if available
		if desc, ok := input["description"].(string); ok && desc != "" {
			return desc
		}
		if cmd, ok := input["command"].(string); ok && cmd != "" {
			return cmd
		}
		b, _ := json.Marshal(input)
		return truncate(string(b), 200)
	}
	return ""
}

func (s *codefreeoSession) handleReasoning(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	if part == nil {
		return
	}
	text, _ := part["text"].(string)
	if text != "" {
		evt := core.Event{Type: core.EventThinking, Content: text}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *codefreeoSession) handleError(raw map[string]any) {
	errMsg := extractErrorMessage(raw)
	slog.Error("codefreeoSession: agent error", "error", errMsg)
	evt := core.Event{Type: core.EventError, Error: fmt.Errorf("%s", errMsg)}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
		return
	}
}

// extractErrorMessage tries to pull a human-readable message from various
// Codefree-O error JSON shapes.
func extractErrorMessage(raw map[string]any) string {
	// Shape: {"error": {"data": {"message": "..."}, "name": "..."}}
	if errObj, ok := raw["error"].(map[string]any); ok {
		if data, ok := errObj["data"].(map[string]any); ok {
			if msg, ok := data["message"].(string); ok && msg != "" {
				name, _ := errObj["name"].(string)
				if name != "" {
					return name + ": " + msg
				}
				return msg
			}
		}
		if msg, ok := errObj["message"].(string); ok && msg != "" {
			return msg
		}
		if name, ok := errObj["name"].(string); ok && name != "" {
			return name
		}
	}
	// Shape: {"error": "string message"}
	if errStr, ok := raw["error"].(string); ok && errStr != "" {
		return errStr
	}
	// Shape: {"part": {"error": "...", "message": "..."}}
	if part, ok := raw["part"].(map[string]any); ok {
		if msg, ok := part["error"].(string); ok && msg != "" {
			return msg
		}
		if msg, ok := part["message"].(string); ok && msg != "" {
			return msg
		}
	}
	if msg, ok := raw["message"].(string); ok && msg != "" {
		return msg
	}
	b, _ := json.Marshal(raw)
	return string(b)
}

func (s *codefreeoSession) handleStepStart(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	if part == nil {
		return
	}
	sessionID, _ := part["sessionID"].(string)
	if sessionID != "" {
		s.chatID.Store(sessionID)
		slog.Debug("codefreeoSession: session started", "session_id", sessionID)
	}
}

func (s *codefreeoSession) handleStepFinish(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	reason, _ := part["reason"].(string)
	slog.Debug("codefreeoSession: step finished", "reason", reason, "session_id", s.CurrentSessionID())
}

// RespondPermission is a no-op — Codefree-O handles permissions internally.
func (s *codefreeoSession) RespondPermission(_ string, _ core.PermissionResult) error {
	return nil
}

func (s *codefreeoSession) Events() <-chan core.Event {
	return s.events
}

func (s *codefreeoSession) CurrentSessionID() string {
	v, _ := s.chatID.Load().(string)
	return v
}

func (s *codefreeoSession) Alive() bool {
	return s.alive.Load()
}

func (s *codefreeoSession) Close() error {
	s.alive.Store(false)
	s.cancel()
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(8 * time.Second):
		slog.Warn("codefreeoSession: close timed out, abandoning wg.Wait")
	}
	close(s.events)
	return nil
}

func truncate(s string, maxRunes int) string {
	if utf8.RuneCountInString(s) <= maxRunes {
		return s
	}
	return string([]rune(s)[:maxRunes]) + "..."
}
