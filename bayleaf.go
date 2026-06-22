package bayleaf

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"time"
)

const ReleaseTag = "v0.1.0"

const DefaultModel = "intfloat/e5-base"

// Config controls how bayleaf manages the Python subprocess.
type Config struct {
	// Where downloaded runtime + models are cached. Default: ~/.bayleaf/cache/
	CacheDir string

	// Path for the Unix domain socket. Default: <CacheDir>/bayleaf.sock
	SocketPath string

	// Seconds of inactivity before the server shuts down.
	// Default: 300 (5 minutes). Set to -1 to disable idle shutdown (always running).
	IdleTimeout int

	// Logger for lifecycle events. Default: discard.
	Logger *log.Logger
}

// Client communicates with the bayleaf Python server.
type Client struct {
	srv    *server
	config Config
}

// Start initializes the client. It does NOT start the Python server --
// the server boots lazily on first Embed/Convert call, or eagerly via Warm().
func Start(ctx context.Context, cfg Config) (*Client, error) {
	if cfg.CacheDir == "" {
		cfg.CacheDir = defaultCacheDir()
	}
	if cfg.SocketPath == "" {
		cfg.SocketPath = filepath.Join(cfg.CacheDir, "bayleaf.sock")
	}
	if cfg.Logger == nil {
		cfg.Logger = log.New(io.Discard, "", 0)
	}

	idleTimeout := 300 * time.Second
	if cfg.IdleTimeout > 0 {
		idleTimeout = time.Duration(cfg.IdleTimeout) * time.Second
	} else if cfg.IdleTimeout == -1 {
		// Explicitly disable idle shutdown (always running).
		idleTimeout = 0
	}

	runtimeDir, err := ensureRuntime(ctx, cfg.CacheDir, cfg.Logger)
	if err != nil {
		return nil, fmt.Errorf("bayleaf: %w", err)
	}

	srv := newServer(cfg.SocketPath, runtimeDir, idleTimeout, cfg.Logger)

	return &Client{srv: srv, config: cfg}, nil
}

// Embed generates a 768-dimensional embedding vector for the given text.
// Cold-starts the server if not running.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return nil, err
	}

	resp, err := c.srv.doRequest(ctx, http.MethodPost, "/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bayleaf embed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, parseErrorResponse(resp)
	}

	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("bayleaf embed: decoding response: %w", err)
	}
	return result.Embedding, nil
}

// Convert converts a file at the given path to markdown.
// The Python server reads the file directly from disk.
// Cold-starts the server if not running.
func (c *Client) Convert(ctx context.Context, filePath string) (string, error) {
	body, err := json.Marshal(map[string]string{"path": filePath})
	if err != nil {
		return "", err
	}

	resp, err := c.srv.doRequest(ctx, http.MethodPost, "/convert", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("bayleaf convert: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", parseErrorResponse(resp)
	}

	var result struct {
		Markdown string `json:"markdown"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("bayleaf convert: decoding response: %w", err)
	}
	return result.Markdown, nil
}

// Warm boots the Python server in the background without blocking.
// The server starts accepting requests immediately; the embedding
// model loads lazily on first Embed call.
func (c *Client) Warm(ctx context.Context) error {
	go func() {
		if err := c.srv.ensureRunning(ctx); err != nil {
			c.srv.logger.Printf("warm failed: %v", err)
		}
	}()
	return nil
}

// Stop shuts down the Python server immediately.
func (c *Client) Stop() error {
	c.srv.stop()
	return nil
}

// Model returns the name of the embedding model.
func (c *Client) Model() string {
	return DefaultModel
}
