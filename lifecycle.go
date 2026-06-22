package bayleaf

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"
)

type server struct {
	mu         sync.Mutex
	cmd        *exec.Cmd
	socketPath string
	runtimeDir string
	httpClient *http.Client
	logger     interface{ Printf(string, ...any) }

	idleTimeout time.Duration
	lastRequest time.Time
	idleStop    chan struct{}
}

func newServer(socketPath, runtimeDir string, idleTimeout time.Duration, logger interface{ Printf(string, ...any) }) *server {
	transport := &http.Transport{
		DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
			return net.Dial("unix", socketPath)
		},
	}

	return &server{
		socketPath:  socketPath,
		runtimeDir:  runtimeDir,
		httpClient:  &http.Client{Transport: transport, Timeout: 120 * time.Second},
		logger:      logger,
		idleTimeout: idleTimeout,
	}
}

func (s *server) ensureRunning(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cmd != nil && s.isHealthy(ctx) {
		return nil
	}

	if s.cmd != nil {
		s.stopLocked()
	}

	if _, err := os.Stat(s.socketPath); err == nil {
		if !s.isHealthy(ctx) {
			os.Remove(s.socketPath)
		} else {
			return nil
		}
	}

	return s.startLocked(ctx)
}

func (s *server) startLocked(ctx context.Context) error {
	pythonBin := filepath.Join(s.runtimeDir, "python", "bin", "python3")
	serverDir := filepath.Join(s.runtimeDir, "server")
	modelDir := filepath.Join(s.runtimeDir, "models")

	venvSitePackages, err := findSitePackages(s.runtimeDir)
	if err != nil {
		return fmt.Errorf("finding site-packages: %w", err)
	}

	cmd := exec.CommandContext(ctx,
		pythonBin, "-m", "uvicorn", "main:app", "--uds", s.socketPath,
	)
	cmd.Dir = serverDir
	cmd.Env = append(os.Environ(),
		"BAYLEAF_MODEL_DIR="+modelDir,
		"PYTHONPATH="+venvSitePackages,
		"PYTHONDONTWRITEBYTECODE=1",
	)

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("creating stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting python server: %w", err)
	}

	s.cmd = cmd
	s.logger.Printf("python server started (pid %d)", cmd.Process.Pid)

	go func() {
		out, _ := io.ReadAll(stderr)
		err := cmd.Wait()
		s.mu.Lock()
		if s.cmd == cmd {
			s.cmd = nil
		}
		s.mu.Unlock()
		if err != nil {
			s.logger.Printf("python server exited: %v\n%s", err, out)
		}
	}()

	if err := s.waitForHealth(ctx, 10*time.Second); err != nil {
		s.stopLocked()
		return fmt.Errorf("server failed to become healthy: %w", err)
	}

	s.lastRequest = time.Now()
	if s.idleTimeout > 0 {
		s.idleStop = make(chan struct{})
		go s.idleLoop(s.idleStop)
	}

	return nil
}

func (s *server) waitForHealth(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if s.isHealthy(ctx) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
	return fmt.Errorf("health check timed out after %s", timeout)
}

func (s *server) isHealthy(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://bayleaf/health", nil)
	if err != nil {
		return false
	}
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (s *server) idleLoop(stop chan struct{}) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			s.mu.Lock()
			idle := time.Since(s.lastRequest)
			s.mu.Unlock()

			if idle > s.idleTimeout {
				s.logger.Printf("idle timeout (%s), shutting down server", s.idleTimeout)
				s.stop()
				return
			}
		}
	}
}

func (s *server) resetIdle() {
	s.mu.Lock()
	s.lastRequest = time.Now()
	s.mu.Unlock()
}

func (s *server) stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stopLocked()
}

func (s *server) stopLocked() {
	if s.idleStop != nil {
		close(s.idleStop)
		s.idleStop = nil
	}

	if s.cmd == nil || s.cmd.Process == nil {
		os.Remove(s.socketPath)
		return
	}

	pid := s.cmd.Process.Pid
	s.logger.Printf("stopping python server (pid %d)", pid)

	s.cmd.Process.Signal(syscall.SIGTERM)

	done := make(chan struct{})
	go func() {
		s.cmd.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		s.cmd.Process.Signal(syscall.SIGKILL)
		<-done
	}

	s.cmd = nil
	os.Remove(s.socketPath)
}

func (s *server) doRequest(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	if err := s.ensureRunning(ctx); err != nil {
		return nil, err
	}

	resp, err := s.rawRequest(ctx, method, path, body)
	if err != nil {
		s.logger.Printf("request failed, restarting server: %v", err)
		s.mu.Lock()
		s.stopLocked()
		s.mu.Unlock()

		if err := s.ensureRunning(ctx); err != nil {
			return nil, fmt.Errorf("restart failed: %w", err)
		}
		resp, err = s.rawRequest(ctx, method, path, body)
		if err != nil {
			return nil, fmt.Errorf("request failed after restart: %w", err)
		}
	}

	s.resetIdle()
	return resp, nil
}

func (s *server) rawRequest(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, "http://bayleaf"+path, body)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return s.httpClient.Do(req)
}

type serverError struct {
	Error  string `json:"error"`
	Detail string `json:"detail"`
}

func parseErrorResponse(resp *http.Response) error {
	defer resp.Body.Close()
	var se serverError
	if err := json.NewDecoder(resp.Body).Decode(&se); err == nil && se.Detail != "" {
		return fmt.Errorf("%s: %s", se.Error, se.Detail)
	}
	return fmt.Errorf("server returned HTTP %d", resp.StatusCode)
}

func findSitePackages(runtimeDir string) (string, error) {
	venvLib := filepath.Join(runtimeDir, "venv", "lib")
	entries, err := os.ReadDir(venvLib)
	if err != nil {
		return "", err
	}
	for _, e := range entries {
		if e.IsDir() {
			sp := filepath.Join(venvLib, e.Name(), "site-packages")
			if _, err := os.Stat(sp); err == nil {
				return sp, nil
			}
		}
	}
	return "", fmt.Errorf("site-packages not found in %s", venvLib)
}
