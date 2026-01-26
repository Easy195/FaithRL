Model Engine
============



Last updated: 09/25/2025.

Current Support Matrix
----------------------

+----------+-----------+--------------+-------------+--------------------------+
| Backends | Model     | Scalability  | Model       | Pain points              |
|          | Supported |              | Definition  |                          |
|          |           |              |             |                          |
+==========+===========+==============+=============+==========================+
| FSDP     | Day 1     | - Dense is OK| Huggingface | Monkey patch can be      |
| +        | support   |              | + monkey    | easily impacted by       |
| ulysses  | HF model  | - MoE is bad | patch       | transformers version     |
+----------+-----------+--------------+-------------+--------------------------+
| MCore    | Limited   | Best         | GPTModel    | Supporting new models is |
|          |           |              | (One model  | difficult                |
|          |           |              | for all)    |                          |
+----------+-----------+--------------+-------------+--------------------------+

-  We monkey patch attention function to support ulysses
-  We monkey patch VLM models to support FSDP with mixed data with and
   without images

Class Hierarchy
---------------

Note that all the workers and trainers run in **SPMD** mode. SFT/DPO/RM
trainer is directly invoked by ``torchrun``. The Actor/Critic worker can
also be invoked by a RayWorkerGroup and provides APIs to a single
controller.

-  Base Engine level: implement model init, optimizer init, lr scheduler
   init, sharding, checkpoint manager.
-  Full Engine level: subclass base engine and implement
   ``forward_step``.
-  Worker/SPMD trainer level: **engine agnostic**, implement training
   logics using abstract engine APIs

RL trainer utilizes workers to construct HybridFlow program. This is out
of the scope of model engine.

Existing Model Types
--------------------

========== ====================== ======================
Model type Language model         Value model
========== ====================== ======================
Input      text/image/video/audio text/image/video/audio
Output     logits for next token  logits as value
========== ====================== ======================

Currently, we have two model types: language model and value model. We
expect to expand the category to include Qwen-Omni family (output both
text and audio) and VLA models.

Data Format
-----------

Currently, verl adopts left-right padding data format in RL trainer.
This creates massive padding when the discrepancy between response
length is large. We will start to implement no-padding format throughout
the whole system.

   :alt: Data Format

Here is the migration plan:
- Implement no-padding format in engine
- Add a transformation layer in Actor/Critic worker.
- Replace Actor/Critic Worker in RL trainer
- Implement no-padding throughput system

Checkpoint System
-----------------

   :alt: Model Engine Checkpoint System

The engine constructs the model using huggingface config, then load
weights from huggingface checkpoint. If the engine directly uses
huggingface model definition, it can use function provided by
``transformers``. Otherwise, each engine has to write their own
checkpoint load logic (e.g.,
training, each engine has to implement save_checkpoint and
load_checkpoint that save/load intermediate sharded checkpoint including
model, optimizer and lr scheduler states. Each engine has to implement a
checkpoint merge script, that merges the intermediate sharded checkpoint
back to huggingface format.

API
---

A tentative model engine API can be found:

Extension
---------

Add a new backend
~~~~~~~~~~~~~~~~~

-  Start a new folder under ``verl/workers/engine``. Then, implement
   ``transformer_impl.py``. If you want to implement a non-transformer
   model, please contact us in advance.
-  Add the engine config to the GSM8k SFT trainer script:
-  Invoke the tests with your backend:
   This test script will run various backends and various
   configurations, and compare the loss and grad norm of the first step
   to make sure they are close.

Add a new model type
~~~~~~~~~~~~~~~~~~~~

-  This is mainly reserved for models whose the output is not just text
   (e.g., Qwen3-Omni). Please discuss with us before you proceed.


