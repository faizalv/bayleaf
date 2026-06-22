package bayleaf

import (
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestModel(t *testing.T) {
	cfg := Config{
		SocketPath: filepath.Join(t.TempDir(), "bayleaf.sock"),
		Logger:     log.New(os.Stderr, "[test] ", log.LstdFlags),
	}
	client := &Client{
		srv:    newServer(cfg.SocketPath, "", 0, cfg.Logger),
		config: cfg,
	}

	if got := client.Model(); got != DefaultModel {
		t.Errorf("Model() = %q, want %q", got, DefaultModel)
	}
}

func TestDetectPlatform(t *testing.T) {
	platform, err := detectPlatform()
	if err != nil {
		t.Fatalf("detectPlatform() on %s/%s: %v", runtime.GOOS, runtime.GOARCH, err)
	}

	if !strings.Contains(platform, runtime.GOOS) {
		t.Errorf("platform %q does not contain OS %q", platform, runtime.GOOS)
	}
	if !strings.Contains(platform, runtime.GOARCH) {
		t.Errorf("platform %q does not contain arch %q", platform, runtime.GOARCH)
	}
}

func TestTarballURL(t *testing.T) {
	url := tarballURL("linux-amd64")
	want := "https://github.com/faizalv/bayleaf/releases/download/" + ReleaseTag + "/linux-amd64.tar.gz"
	if url != want {
		t.Errorf("tarballURL = %q, want %q", url, want)
	}
}

func TestChecksumURL(t *testing.T) {
	url := checksumURL("darwin-arm64")
	want := "https://github.com/faizalv/bayleaf/releases/download/" + ReleaseTag + "/darwin-arm64.tar.gz.sha256"
	if url != want {
		t.Errorf("checksumURL = %q, want %q", url, want)
	}
}

func TestDefaultCacheDir(t *testing.T) {
	dir := defaultCacheDir()
	if dir == "" {
		t.Fatal("defaultCacheDir returned empty string")
	}

	home, err := os.UserHomeDir()
	if err == nil {
		want := filepath.Join(home, ".bayleaf", "cache")
		if dir != want {
			t.Errorf("defaultCacheDir = %q, want %q", dir, want)
		}
	}
}

func TestVerifyChecksum(t *testing.T) {
	dir := t.TempDir()
	dataFile := filepath.Join(dir, "test.dat")
	checksumFile := filepath.Join(dir, "test.dat.sha256")

	os.WriteFile(dataFile, []byte("hello world\n"), 0644)
	os.WriteFile(checksumFile, []byte("a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"), 0644)

	if !verifyChecksum(dataFile, checksumFile) {
		t.Error("expected checksum to pass")
	}

	os.WriteFile(checksumFile, []byte("0000000000000000000000000000000000000000000000000000000000000000"), 0644)
	if verifyChecksum(dataFile, checksumFile) {
		t.Error("expected checksum to fail with wrong hash")
	}
}

func TestIsClientError(t *testing.T) {
	tests := []struct {
		err  error
		want bool
	}{
		{&httpError{code: 404, url: "http://example.com"}, true},
		{&httpError{code: 403, url: "http://example.com"}, true},
		{&httpError{code: 500, url: "http://example.com"}, false},
		{&httpError{code: 502, url: "http://example.com"}, false},
		{os.ErrNotExist, false},
	}

	for _, tt := range tests {
		if got := isClientError(tt.err); got != tt.want {
			t.Errorf("isClientError(%v) = %v, want %v", tt.err, got, tt.want)
		}
	}
}
