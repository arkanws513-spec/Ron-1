# Ron-1

Ron-1 is an independent conversational assistant built on the Qwen3-1.7B language model as its underlying intelligence layer.

## الهدف

Ron-1 is designed to hold normal conversations and help with research, writing, analysis, and user-provided documents.

Example:

- User: مرحبًا
- Ron-1: أهلًا بك، كيف يمكنني مساعدتك؟
- User: عندي مقالة عن حرب البسوس وأريد إكمالتها.
- Ron-1: أرسل المقالة، وسأفهم ما كتبته ثم أساعدك في إكمالها والحفاظ على سياقها وأسلوبها.

## Architecture

Qwen3-1.7B provides the base language model. Ron Core sits above it and manages conversation context, persistent memory, task handling, writing workflow, and future tools.

The upstream model is loaded from Hugging Face at runtime; model weights are not stored in this Git repository.

## Model

Default model: Qwen/Qwen3-1.7B

Upstream model: https://huggingface.co/Qwen/Qwen3-1.7B

License: Apache-2.0

## Status

Initial Ron Core implementation. The first goal is a working chat loop before adding more advanced memory, tools, retrieval, and learning components.
