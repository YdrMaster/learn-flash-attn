﻿# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.13] - 2025-07-24

### Add

- 修改 crate 名字，增加对外接口；

## [0.0.12] - 2025-07-22

### Changed

- 更新为 flash attention2；

## [0.0.11] - 2025-07-21

### Changed

- 添加支持分页和边长批次的 kv cache 连接 kernel；
- 使用 rayon 使 CPU 在 block 级别并行计算；

## [0.0.10] - 2025-07-17

### Add

- 添加 flash attention 的 cuda kernel；

## [0.0.9] - 2025-07-16

### Changed

- 重构 flash attention，添加一系列具有 `Clone + Copy + repr(C)` 的结构体，可在 CPU 和 GPU 实现中复用；
- 对 CPU 执行顺序做 shuffle 以验证任意性；

## [0.0.8] - 2025-06-25

### Added

- 支持 paged attention；

## [0.0.7] - 2025-06-25

### Added

- 支持 kv cache；

### Changed

- 不再要求 q kv 分块等大；

## [0.0.6] - 2025-06-24

### Changed

- 根据 cuda flash attention 重构算法，用 CPU 模拟 GPU 并行；

## [0.0.5] - 2025-06-20

### Changed

- 接口支持 GQA；
- 实现 seq 和 ctx 维度分块并行的 flash attention；

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

[Unreleased]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.13...HEAD
[0.0.13]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.12...v0.0.13
[0.0.12]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.11...v0.0.12
[0.0.11]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.10...v0.0.11
[0.0.10]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.9...v0.0.10
[0.0.9]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/YdrMaster/learn-flash-attn/compare/v0.0.0...v0.0.1
[0.0.0]: https://github.com/YdrMaster/learn-flash-attn/releases/tag/v0.0.0
