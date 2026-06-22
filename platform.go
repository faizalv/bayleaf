package bayleaf

import (
	"fmt"
	"runtime"
)

func detectPlatform() (string, error) {
	os := runtime.GOOS
	arch := runtime.GOARCH

	switch os {
	case "linux", "darwin":
	default:
		return "", fmt.Errorf("unsupported OS: %s", os)
	}

	switch arch {
	case "amd64", "arm64":
	default:
		return "", fmt.Errorf("unsupported architecture: %s", arch)
	}

	return fmt.Sprintf("%s-%s", os, arch), nil
}

func tarballURL(platform string) string {
	return fmt.Sprintf(
		"https://github.com/faizalv/bayleaf/releases/download/%s/%s.tar.gz",
		ReleaseTag, platform,
	)
}

func checksumURL(platform string) string {
	return fmt.Sprintf(
		"https://github.com/faizalv/bayleaf/releases/download/%s/%s.tar.gz.sha256",
		ReleaseTag, platform,
	)
}
