//go:build integration

package bayleaf

import (
	"context"
	"log"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func integrationConfig(t *testing.T) Config {
	t.Helper()
	return Config{
		CacheDir:    os.Getenv("BAYLEAF_TEST_CACHE"),
		SocketPath:  t.TempDir() + "/bayleaf-test.sock",
		IdleTimeout: 30,
		Logger:      log.New(os.Stderr, "[bayleaf-test] ", log.LstdFlags),
	}
}

func TestIntegrationStartStop(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := Start(ctx, integrationConfig(t))
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	if err := client.Warm(ctx); err != nil {
		t.Fatalf("Warm: %v", err)
	}

	time.Sleep(2 * time.Second)

	if err := client.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
}

func TestIntegrationEmbed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := Start(ctx, integrationConfig(t))
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Stop()

	vec, err := client.Embed(ctx, "query: what is machine learning")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(vec) != 768 {
		t.Errorf("expected 768-dim vector, got %d", len(vec))
	}

	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("expected L2 norm ~1.0, got %f", norm)
	}
}

func TestIntegrationConvert(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := Start(ctx, integrationConfig(t))
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Stop()

	tmpFile := t.TempDir() + "/test.txt"
	if err := os.WriteFile(tmpFile, []byte("hello world"), 0644); err != nil {
		t.Fatalf("writing test file: %v", err)
	}

	markdown, err := client.Convert(ctx, tmpFile)
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	if markdown == "" {
		t.Error("expected non-empty markdown output")
	}
}

func TestIntegrationColdStart(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := Start(ctx, integrationConfig(t))
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Stop()

	vec, err := client.Embed(ctx, "query: cold start test")
	if err != nil {
		t.Fatalf("Embed (cold start): %v", err)
	}

	if len(vec) != 768 {
		t.Errorf("expected 768-dim vector, got %d", len(vec))
	}
}

func TestIntegrationIdleShutdown(t *testing.T) {
	cfg := integrationConfig(t)
	cfg.IdleTimeout = 3

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := Start(ctx, cfg)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Stop()

	_, err = client.Embed(ctx, "query: idle test")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	time.Sleep(5 * time.Second)

	if _, err := os.Stat(cfg.SocketPath); err == nil {
		t.Error("socket file still exists after idle timeout")
	}

	vec, err := client.Embed(ctx, "query: after idle restart")
	if err != nil {
		t.Fatalf("Embed after idle restart: %v", err)
	}
	if len(vec) != 768 {
		t.Errorf("expected 768-dim vector, got %d", len(vec))
	}
}

func TestIntegrationConcurrentColdStart(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := Start(ctx, integrationConfig(t))
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Stop()

	var wg sync.WaitGroup
	var failures atomic.Int32

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			vec, err := client.Embed(ctx, "query: concurrent test")
			if err != nil {
				t.Errorf("concurrent Embed: %v", err)
				failures.Add(1)
				return
			}
			if len(vec) != 768 {
				t.Errorf("expected 768-dim, got %d", len(vec))
				failures.Add(1)
			}
		}()
	}

	wg.Wait()
	if f := failures.Load(); f > 0 {
		t.Errorf("%d/%d concurrent requests failed", f, 10)
	}
}

func TestIntegrationCrashRecovery(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := Start(ctx, integrationConfig(t))
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Stop()

	_, err = client.Embed(ctx, "query: before crash")
	if err != nil {
		t.Fatalf("Embed before crash: %v", err)
	}

	client.srv.mu.Lock()
	if client.srv.cmd != nil && client.srv.cmd.Process != nil {
		client.srv.cmd.Process.Kill()
	}
	client.srv.cmd = nil
	client.srv.mu.Unlock()

	time.Sleep(500 * time.Millisecond)

	vec, err := client.Embed(ctx, "query: after crash")
	if err != nil {
		t.Fatalf("Embed after crash: %v", err)
	}
	if len(vec) != 768 {
		t.Errorf("expected 768-dim vector, got %d", len(vec))
	}
}
