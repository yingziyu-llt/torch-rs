//! # torch-rs
//!
//! 一个简易的神经网络/自动微分库，支持张量、自动求导、基础神经网络层与优化器。
//!
//! - 张量结构体与操作
//! - 自动求导机制
//! - 线性层、激活层等神经网络模块
//! - SGD等优化器
//! - 兼容ndarray
//! 由于项目开发时间较短，功能较为基础，主要用于学习，还存在很多问题，如数值不稳定，数据处理接口较为简陋等

pub mod autograd;
pub mod nn;
pub mod ops;
pub mod optimizer;
pub mod tensor;
pub mod utils;
pub mod functional;
