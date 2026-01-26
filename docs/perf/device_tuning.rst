Hardware Resource Needed for RL
===============================

Last updated: 06/25/2025.

Since RL requires more resources compared to regular training, 
determining how much resources are needed to successfully run it before training 
is a relatively difficult task. To provide more people with reference points for 
resource selection when dealing with different models and tasks, this section is 
mainly dedicated to introducing the environmental requirements based on experiments 
we have conducted.

However, due to limited staff and equipment resources, we also hope for more 
contributions from the open-source community. When submitting a PR, it is necessary 
to provide a script to be added to the example/tuning scripts.

We need two types of scripts: one is the configuration that can run with the **minimum 
resources(min)**, and the other is the configuration that runs with **recommended resources(recommended)**. For the former, 
it can be understood as a script that can run after applying all memory optimization techniques 
(e.g., offload, gradient checkpointing). For the latter, it can be understood as a script that 
can run while avoiding operations that incur additional time overhead as much as possible (targetting best throughput).

When defining script names, please follow this format: 
``[model]_[task]_[gpunums]_[device]_[train]_[infer].sh``. This will effectively improve 
the script's recognizability. You can place the script under the ``examples/tuning/`` directory.

If you happen to have a configuration that has already been tested, we welcome you to submit 
a PR and include a screenshot from Wandb or other verifiable evidence.

----------------------------------------

0.5B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2.5-0.5B
      - GRPO-LoRA
      - 1*H100
      - 116
      - fsdp
      - vllm0.8.3
      - Contributor

1.5B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2.5-1.5B
      - GRPO-LoRA
      - 1*H100
      - 128
      - fsdp
      - vllm0.8.3
      - Contributor

3B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2.5-3B
      - GRPO-LoRA
      - 1*H100
      - 62
      - fsdp
      - vllm0.8.3
      - Contributor

7B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-7B
      - GRPO
      - 2*H800
      - \
      - fsdp
      - vllm0.8.2
      - Contributor
    * - MIN
      - Qwen2.5-7B
      - GRPO-LoRA
      - 1*H100
      - 16
      - fsdp
      - vllm0.8.3
      - Contributor

14B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-14B
      - GRPO
      - 4*H800
      - \
      - fsdp
      - vllm0.8.2
      - Contributor
    * - MIN
      - Qwen2.5-14B
      - GRPO-LoRA
      - 2*H100
      - 116
      - fsdp
      - vllm0.8.3
      - Contributor

32B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-32B
      - GRPO
      - 8*H20
      - \
      - megatron
      - vllm0.8.2
      - Contributor
    * - MIN
      - Qwen2.5-32B
      - GRPO-LoRA
      - 4*H100
      - 180
      - fsdp
      - vllm0.8.3
      - Contributor

70B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-70B
      - GRPO
      - 32*H20
      - \
      - fsdp
      - vllm0.8.2
      - Contributor
    * - MIN
      - Qwen2-70B
      - GRPO
      - 32*H800
      - \
      - fsdp
      - vllm0.8.3
      - Contributor
    * - MIN
      - Qwen2.5-72B
      - GRPO-LoRA
      - 8*H100
      - 176
      - fsdp
      - vllm0.8.3
      - Contributor

405B
~~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ======== ====== ====== ======
   tag    model  task   resource MaxBatch train  infer  link
   ====== ====== ====== ======== ======== ====== ====== ======
   \      \      \        \        \      \      \
   ====== ====== ====== ======== ======== ====== ====== ======

671B
~~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ======== ====== ====== ======
   tag    model  task   resource MaxBatch train  infer  link
   ====== ====== ====== ======== ======== ====== ====== ======
   \      \      \        \        \      \      \
   ====== ====== ====== ======== ======== ====== ====== ======

