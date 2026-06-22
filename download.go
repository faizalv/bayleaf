package bayleaf

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/faizalv/bayleaf/internal/archive"
)

func defaultCacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "bayleaf", "cache")
	}
	return filepath.Join(home, ".bayleaf", "cache")
}

// EnsureRuntime downloads and extracts the Python runtime + model weights
// if not already cached. Returns the path to the extracted runtime directory.
func EnsureRuntime(ctx context.Context, cacheDir string) (string, error) {
	return ensureRuntime(ctx, cacheDir, log.New(io.Discard, "", 0))
}

func ensureRuntime(ctx context.Context, cacheDir string, logger *log.Logger) (string, error) {
	if cacheDir == "" {
		cacheDir = defaultCacheDir()
	}

	runtimeDir := filepath.Join(cacheDir, ReleaseTag)
	sentinelPath := filepath.Join(runtimeDir, "server", "main.py")

	if _, err := os.Stat(sentinelPath); err == nil {
		return runtimeDir, nil
	}

	// Clean up stale extraction attempts.
	extractingDir := filepath.Join(cacheDir, ".extracting")
	os.RemoveAll(extractingDir)

	platform, err := detectPlatform()
	if err != nil {
		return "", err
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", fmt.Errorf("creating cache dir: %w", err)
	}

	tarballPath := filepath.Join(cacheDir, fmt.Sprintf("%s-%s.tar.gz", ReleaseTag, platform))
	checksumPath := tarballPath + ".sha256"

	if err := downloadWithRetry(ctx, checksumURL(platform), checksumPath, logger); err != nil {
		return "", fmt.Errorf("downloading checksum: %w", err)
	}

	if !verifyChecksum(tarballPath, checksumPath) {
		os.Remove(tarballPath)
		os.Remove(tarballPath + ".partial")

		if err := downloadResumable(ctx, tarballURL(platform), tarballPath, logger); err != nil {
			return "", fmt.Errorf("downloading tarball: %w", err)
		}

		if !verifyChecksum(tarballPath, checksumPath) {
			os.Remove(tarballPath)
			return "", fmt.Errorf("checksum mismatch after download")
		}
	}

	logger.Printf("extracting runtime to %s", runtimeDir)
	if err := os.MkdirAll(extractingDir, 0755); err != nil {
		return "", fmt.Errorf("creating extraction dir: %w", err)
	}

	if err := archive.ExtractTarGz(tarballPath, extractingDir); err != nil {
		os.RemoveAll(extractingDir)
		os.Remove(tarballPath)
		return "", fmt.Errorf("extracting tarball: %w", err)
	}

	os.RemoveAll(runtimeDir)
	if err := os.Rename(extractingDir, runtimeDir); err != nil {
		os.RemoveAll(extractingDir)
		return "", fmt.Errorf("finalizing extraction: %w", err)
	}

	os.Remove(tarballPath)
	os.Remove(tarballPath + ".partial")

	logger.Printf("runtime ready at %s", runtimeDir)
	return runtimeDir, nil
}

func verifyChecksum(filePath, checksumPath string) bool {
	expected, err := os.ReadFile(checksumPath)
	if err != nil {
		return false
	}

	f, err := os.Open(filePath)
	if err != nil {
		return false
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false
	}

	got := hex.EncodeToString(h.Sum(nil))
	return strings.TrimSpace(string(expected)) == got
}

func downloadResumable(ctx context.Context, url, dest string, logger *log.Logger) error {
	partialPath := dest + ".partial"

	var startByte int64
	if info, err := os.Stat(partialPath); err == nil {
		startByte = info.Size()
	}

	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<(attempt-1)*2+1) * time.Second // 1s, 3s, 9s
			logger.Printf("retry %d/%d in %s", attempt+1, 3, backoff)

			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		err := doDownload(ctx, url, partialPath, startByte)
		if err == nil {
			return os.Rename(partialPath, dest)
		}

		if isClientError(err) {
			return err
		}

		logger.Printf("download attempt %d failed: %v", attempt+1, err)

		if info, statErr := os.Stat(partialPath); statErr == nil {
			startByte = info.Size()
		}
	}

	return fmt.Errorf("download failed after 3 attempts")
}

func doDownload(ctx context.Context, url, dest string, startByte int64) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}

	if token := os.Getenv("GITHUB_TOKEN"); token != "" {
		req.Header.Set("Authorization", "token "+token)
	}
	if startByte > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return &httpError{code: resp.StatusCode, url: url}
	}

	flags := os.O_CREATE | os.O_WRONLY
	if resp.StatusCode == http.StatusPartialContent {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}

	f, err := os.OpenFile(dest, flags, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	return err
}

func downloadWithRetry(ctx context.Context, url, dest string, logger *log.Logger) error {
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<(attempt-1)*2+1) * time.Second
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		err := doDownload(ctx, url, dest, 0)
		if err == nil {
			return nil
		}

		if isClientError(err) {
			return err
		}

		logger.Printf("download attempt %d failed: %v", attempt+1, err)
	}
	return fmt.Errorf("download failed after 3 attempts")
}

type httpError struct {
	code int
	url  string
}

func (e *httpError) Error() string {
	return fmt.Sprintf("HTTP %d from %s", e.code, e.url)
}

func isClientError(err error) bool {
	if he, ok := err.(*httpError); ok {
		return he.code >= 400 && he.code < 500
	}
	return false
}
