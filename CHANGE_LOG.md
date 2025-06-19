# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- 支持 GQA；
- 对 seq 和 ctx 维度分块；

## [0.0.4] - 2025-06-19

### Added

- 添加 attention 实现和测试框架；
- 实现基本的 flash attention；

## [0.0.3] - 2025-06-19

### Changed

- 改为真正的分块 online softmax，并尽量用比较判断替换 exp；

## [0.0.2] - 2025-06-18

### Changed

- 改为标准的 block online softmax，但是没有做分块；

## [0.0.1] - 2025-06-18

### Changed

- online softmax 用判断替换一次 exp；

## [0.0.0] - 2025-06-18

### Added

- 创建项目；
- 实现基本的 online softmax；

[Unreleased]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.0...v0.0.1
[0.0.0]: https://github.com/YdrMaster/learn-flash-attn/releases/tag/v0.0.0
