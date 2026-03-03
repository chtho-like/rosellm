#!/bin/bash
codex --search exec --dangerously-bypass-approvals-and-sandbox  \
  -m gpt-5.2 -c model_reasoning_effort="xhigh" \
  '在 gemm_cuda 文件夹下面做这样的事情,穷尽你所有的可能,但是不允许使用 tensor core 相关,尝试一步步优化出比 cuBLAS 还要高性能的 SGEMM 算子,FP32 的那个,cuBLAS 内部实现 SGEMM 的时候应该也没用 tensor core,所以你也不应该用 tensor core,你需要穷尽所有可能探索,然后一步步地进行优化,每个优化对应一个文件,相邻两个优化文件之间的 diff 尽可能清晰易读,总优化次数不设上限,要求每次优化和 cuBLAS 的性能结果进行对比画图,可以参考 gemm2,但是你应该尽可能地科学合理,符合业界的常用 benchmark 形式,并详细记录所有实验命令和实验结果以及对应的详细分析在一份报告 md 中,最终要求你写出来的性能至少要达到乃至超越 cuBLAS 的水平,只需处理当前机器上的这张卡就行了,你可以在互联网上找任何资料,可以供参考的是 https://salykova.github.io/gemm-gpu 以及里面提到的所有链接 github blog 等等,你做优化如果参考到了任何的实现,可以给出引用来源,你不要拘泥于任何东西,你需要打破常规进行创新,来使你写的算子性能达到无法匹及的程度,你可以使用任何工具，比如 ncu 等,你可以在报告中详细给出分析数据以及对应的方案和验证等,可以无比细致地详细'

codex --search exec --dangerously-bypass-approvals-and-sandbox  \
  -m gpt-5.2 -c model_reasoning_effort="xhigh" \
  '在 gemm_tensor 文件夹下面做这样的事情,穷尽你所有的可能,使用 tensor core 相关,尝试一步步优化出比 cuBLAS 还要高性能的 HGEMM 算子,Half precision 的那个,cuBLAS 内部实现 HGEMM 的时候应该也用 tensor core,所以你也应该用 tensor core,你需要穷尽所有可能探索,然后一步步地进行优化,每个优化对应一个文件,相邻两个优化文件之间的 diff 尽可能清晰易读,总优化次数不设上限,要求每次优化和 cuBLAS 的性能结果进行对比画图,可以参考 gemm2,但是你应该尽可能地科学合理,符合业界的常用 benchmark 形式,并详细记录所有实验命令和实验结果以及对应的详细分析在一份报告 md 中,最终要求你写出来的性能至少要达到乃至超越 cuBLAS 的水平,只需处理当前机器上的这张卡就行了,你可以在互联网上找任何资料,可以供参考的是 https://salykova.github.io/gemm-gpu 以及里面提到的所有链接 github blog 等等,你做优化如果参考到了任何的实现,可以给出引用来源,你不要拘泥于任何东西,你需要打破常规进行创新,来使你写的算子性能达到无法匹及的程度,你可以使用任何工具，比如 ncu 等,你可以在报告中详细给出分析数据以及对应的方案和验证等,可以无比细致地详细'

codex --search exec --dangerously-bypass-approvals-and-sandbox  \
  -m gpt-5.2 -c model_reasoning_effort="xhigh" \
  '在 flashattn 文件夹下面做这样的事情,穷尽你所有的可能,尝试一步步优化出比 著名业界 FlashAttention 仓库(最高性能的那种) 还要高性能的 attention 算子,做你需要穷尽所有可能探索,然后一步步地进行优化,每个优化对应一个文件,相邻两个优化文件之间的 diff 尽可能清晰易读,总优化次数不设上限,要求每次优化和 业界超高性能的 FlashAttention 仓库 的性能结果进行对比画图,可以参考 gemm2,但是你应该尽可能地科学合理,符合业界的常用 benchmark 形式,并详细记录所有实验命令和实验结果以及对应的详细分析在一份报告 md 中,最终要求你写出来的性能至少要达到乃至超越 cuBLAS 的水平,只需处理当前机器上的这张卡就行了,你可以在互联网上找任何资料,你可以自己 git clone 你认为需要的 git 仓库进行源码分析,可以先在 /data/projects /data2/projects 下面看看本机上的一些仓库,没有的时候你再下载,你做优化如果参考到了任何的实现,可以给出引用来源,你不要拘泥于任何东西,你需要打破常规进行创新,来使你写的算子性能达到无法匹及的程度,你可以使用任何工具，比如 ncu 等,你可以在报告中详细给出分析数据以及对应的方案和验证等,可以无比细致地详细'
