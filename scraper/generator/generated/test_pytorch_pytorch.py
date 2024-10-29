import sys
_module = sys.modules[__name__]
del sys
create_test_cert = _module
numpy = _module
compare_with_baseline = _module
get_stats = _module
update_commit_hash = _module
print_sccache_log = _module
run_python_nn_smoketests = _module
normalize_yaml_fragment = _module
trigger_azure_pipeline = _module
build_triton_wheel = _module
check_labels = _module
cherry_pick = _module
close_nonexistent_disable_issues = _module
collect_ciflow_labels = _module
comment_on_pr = _module
convert_lintrunner_annotations_to_github = _module
delete_old_branches = _module
ensure_actions_will_cancel = _module
export_pytorch_labels = _module
file_io_utils = _module
filter_test_configs = _module
generate_binary_build_matrix = _module
generate_ci_workflows = _module
generate_docker_release_matrix = _module
generate_pytorch_version = _module
get_aws_session_tokens = _module
get_workflow_job_id = _module
github_utils = _module
gitutils = _module
label_utils = _module
lint_native_functions = _module
parse_ref = _module
pytest_cache = _module
pytest_caching_utils = _module
runner_determinator = _module
tag_docker_images_for_release = _module
test_check_labels = _module
test_filter_test_configs = _module
test_gitutils = _module
test_label_utils = _module
test_pytest_caching_utils = _module
test_runner_determinator = _module
test_trymerge = _module
test_tryrebase = _module
trymerge = _module
trymerge_explainer = _module
tryrebase = _module
update_runner_determinator = _module
generate_test_torchscripts = _module
make_assets = _module
make_assets_custom = _module
configure = _module
generate_kernels = _module
codegen = _module
benchmark = _module
diff = _module
DummyData = _module
data = _module
launcher = _module
CPUMetric = _module
CUDAMetric = _module
MetricBase = _module
MetricsLogger = _module
ProcessedMetricsPrinter = _module
DummyModel = _module
models = _module
server = _module
trainer = _module
criterions = _module
ddp_models = _module
hook_states = _module
hooks = _module
iteration_steps = _module
preprocess_data = _module
utils = _module
agent = _module
coordinator = _module
observer = _module
dynamo = _module
benchmarks = _module
check_accuracy = _module
check_csv = _module
check_graph_breaks = _module
check_memory_compression_ratio = _module
check_perf_csv = _module
update_expected = _module
combine_csv = _module
common = _module
dist_util = _module
distributed = _module
huggingface = _module
join_results = _module
microbenchmarks = _module
analyze_templates = _module
bench_mm_fusion = _module
benchmark_helper = _module
cache_debug_microbenchmarks = _module
cache_hit_microbenchmarks = _module
dynamo_microbenchmarks = _module
fx_microbenchmarks = _module
inductor_bmm = _module
inductor_cpu_atomic = _module
inductor_mm = _module
matmul_relu = _module
microbench = _module
model = _module
operator_inp_utils = _module
operatorbench = _module
overheads = _module
tensor_layout_mini_benchmark = _module
parse_logs = _module
pr_time_benchmarks = _module
benchmark_base = _module
add_loop = _module
aotdispatcher = _module
aotdispatcher_partitioner = _module
basic_modules_benchmarks = _module
sum_floordiv = _module
symint_sum = _module
update_hint_benchmark = _module
check_results = _module
log_benchmarking_time = _module
runner = _module
summarize_perf = _module
test = _module
timm_models = _module
torchao_backend = _module
torchbench = _module
training_loss = _module
fastrnns = _module
bench = _module
cells = _module
conftest = _module
custom_lstms = _module
factory = _module
fuser = _module
profile = _module
scratch = _module
test_bench = _module
SimpleAddModule = _module
framework_overhead_benchmark = _module
pt_wrapper_module = _module
audio_text_models = _module
compare = _module
functional_autograd_benchmark = _module
ppl_models = _module
torchaudio_models = _module
torchvision_models = _module
vision_models = _module
plot_speedups = _module
run_benchmarks = _module
generate = _module
mixtral_moe_model = _module
mixtral_moe_quantize = _module
quantize = _module
process_metrics = _module
applications = _module
ci = _module
core = _module
api = _module
expand = _module
types = _module
definitions = _module
setup = _module
standard = _module
execution = _module
work = _module
main = _module
worker = _module
nested_bmm_bench = _module
operator_benchmark = _module
benchmark_all_other_test = _module
benchmark_all_quantized_test = _module
benchmark_all_test = _module
benchmark_core = _module
benchmark_pytorch = _module
benchmark_runner = _module
benchmark_test_generator = _module
benchmark_utils = _module
repeat_benchmark = _module
add_ops_list_test = _module
jit_forward_test = _module
pt_backward_test = _module
pt_configs_list_test = _module
pt_cpu_gpu_forward_backward_test = _module
random_sample_test = _module
pt = _module
add_test = _module
ao_sparsifier_test = _module
as_strided_test = _module
batchnorm_test = _module
binary_test = _module
bmm_test = _module
cat_test = _module
channel_shuffle_test = _module
chunk_test = _module
clip_ranges_test = _module
configs = _module
conv_test = _module
diag_test = _module
embeddingbag_test = _module
fill_test = _module
gather_test = _module
gelu_test = _module
groupnorm_test = _module
hardsigmoid_test = _module
hardswish_test = _module
index_select_test = _module
instancenorm_test = _module
interpolate_test = _module
layernorm_test = _module
linear_prepack_fp16_test = _module
linear_test = _module
linear_unpack_fp16_test = _module
matmul_test = _module
matrix_mult_test = _module
nan_to_num_test = _module
pool_test = _module
qactivation_test = _module
qarithmetic_test = _module
qatembedding_ops_test = _module
qbatchnorm_test = _module
qcat_test = _module
qcomparators_test = _module
qconv_test = _module
qembedding_bag_lookups_test = _module
qembedding_pack_test = _module
qembeddingbag_test = _module
qgroupnorm_test = _module
qinstancenorm_test = _module
qinterpolate_test = _module
qlayernorm_test = _module
qlinear_test = _module
qobserver_test = _module
qpool_test = _module
qrnn_test = _module
qtensor_method_test = _module
quantization_test = _module
qunary_test = _module
remainder_test = _module
softmax_test = _module
split_test = _module
stack_test = _module
sum_test = _module
tensor_to_test = _module
unary_test = _module
cpp_extension_test = _module
pyspybench = _module
profiler_bench = _module
resnet_memory_profiler = _module
record_function_bench = _module
nested_annotation_str = _module
simple_measurement = _module
sparse = _module
benchmark_semi_structured_sparsity = _module
dlmc = _module
matmul_bench = _module
spmm = _module
spmv = _module
triton_ops = _module
attention = _module
broadcast = _module
concat = _module
conv = _module
elementwise = _module
matmul = _module
normalization = _module
pooling = _module
pt_engine = _module
reduction = _module
rnn_eltwise = _module
softmax = _module
swish = _module
tensor_engine = _module
attention_bias_benchmarks = _module
better_transformer_vs_mha_functional = _module
score_mod = _module
sdp = _module
sdpa = _module
upload_scribe = _module
hp_emblookup_codegen = _module
sve_emblookup_codegen = _module
conf = _module
build_activation_images = _module
build_opsets = _module
build_quantization_configs = _module
generate_example_rst = _module
build_onnx_dynamo_diagnostics_rules_md = _module
build_onnx_torchscript_supported_aten_op_csv_table = _module
functorch = _module
_src = _module
aot_autograd = _module
eager_transforms = _module
make_functional = _module
vmap = _module
chrome_trace_parser = _module
cse = _module
operator_authoring = _module
per_sample_grads = _module
pointwise_scorecard = _module
process_scorecard = _module
compile = _module
dim = _module
batch_tensor = _module
delayed_mul_tensor = _module
magic_trace = _module
op_properties = _module
reference = _module
tree_map = _module
wrap_type = _module
einops = _module
_parsing = _module
rearrange = _module
eager_fusion = _module
fuse_module = _module
linear_train = _module
simple_function = _module
cifar10_opacus = _module
cifar10_transforms = _module
parallel_train = _module
lennard_jones = _module
omniglot_loaders = _module
evjang = _module
evjang_transforms = _module
evjang_transforms_module = _module
experimental = _module
control_flow = _module
ops = _module
plot_ensembling = _module
plot_jacobians_and_hessians = _module
plot_per_sample_gradients = _module
gen_data = _module
coreml_backend = _module
trace_model = _module
custom_build = _module
run_on_aws_devicefarm = _module
check_mypy_version = _module
sympy_mypy_plugin = _module
format_test_csv = _module
download_reports = _module
failures_histogram = _module
passrate = _module
update_failures = _module
diagnose_protobuf = _module
update_schema = _module
get_python_cmake_flags = _module
log_extract = _module
apply_categories = _module
categorize = _module
classifier = _module
commitlist = _module
namespace_check = _module
test_release_notes = _module
_test_bazel = _module
test_activation_sparsifier = _module
test_composability = _module
test_data_scheduler = _module
test_data_sparsifier = _module
test_kernels = _module
test_parametrization = _module
test_qlinear_packed_params = _module
test_scheduler = _module
test_sparsifier = _module
test_sparsity_utils = _module
test_structured_sparsifier = _module
test_complex = _module
test_functional = _module
test_logging = _module
test_launch = _module
test_benchmark_utils = _module
test_args = _module
test_cuda = _module
cpp = _module
compile_model = _module
init_baseline = _module
optim_baseline = _module
jit = _module
tests_setup = _module
cpp_api_parity = _module
functional_impl_check = _module
module_impl_check = _module
parity_table_parser = _module
sample_functional = _module
sample_module = _module
pytorch_openreg = _module
_aten_impl = _module
_device_daemon = _module
_meta_parser = _module
test_openreg = _module
torch_test_cpp_extension = _module
create_dummy_torchscript_model = _module
backend = _module
test_custom_backend = _module
my_custom_ops = _module
my_custom_ops2 = _module
pointwise = _module
test_custom_ops = _module
test_infer_schema_annotation = _module
delete = _module
test_fully_shard_autograd = _module
test_fully_shard_clip_grad_norm_ = _module
test_fully_shard_comm = _module
test_fully_shard_compile = _module
test_fully_shard_extensions = _module
test_fully_shard_frozen = _module
test_fully_shard_grad_scaler = _module
test_fully_shard_init = _module
test_fully_shard_logging = _module
test_fully_shard_memory = _module
test_fully_shard_mixed_precision = _module
test_fully_shard_overlap = _module
test_fully_shard_state = _module
test_fully_shard_state_dict = _module
test_fully_shard_training = _module
test_fully_shard_compile = _module
test_fully_shard_model_checkpoint = _module
test_fully_shard_optim_checkpoint = _module
test_fully_shard_runtime = _module
test_fully_shard_util = _module
test_checkpoint = _module
test_2d_composability = _module
test_pp_composability = _module
test_compose = _module
test_contract = _module
test_replicate = _module
test_replicate_with_compiler = _module
test_sharded_optim = _module
test_binary_cmp = _module
test_embedding = _module
test_embedding_bag = _module
test_init = _module
test_tensor_ops = _module
test_logger = _module
test_sharded_tensor = _module
test_sharded_tensor_reshard = _module
test_sharding_plan = _module
test_sharding_spec = _module
test_sharder = _module
_tensor = _module
test_comm_mode = _module
test_comm_mode_features = _module
test_op_coverage = _module
test_local_map = _module
test_register_sharding = _module
test_tp_transform = _module
test_api = _module
test_attention = _module
test_common_rules = _module
test_convolution_ops = _module
test_dtensor = _module
test_dtensor_compile = _module
test_dtensor_ops = _module
test_embedding_ops = _module
test_experimental_ops = _module
test_math_ops = _module
test_matrix_ops = _module
test_op_strategy = _module
test_optimizers = _module
test_pointwise_ops = _module
test_random_ops = _module
test_redistribute = _module
test_utils = _module
test_view_ops = _module
test_xla_integration = _module
test_fsdp2_mem_tracker = _module
test_mem_tracker = _module
test_memory_tracker = _module
test_mod_tracker = _module
test_runtime_estimator = _module
test_sac_estimator = _module
test_sac_ilp = _module
test_ddp_hooks = _module
test_quantization = _module
test_join = _module
argparse_util_test = _module
test_script = _module
test_e2e_save_and_load = _module
test_fine_tuning = _module
test_fsdp_ep = _module
test_pipeline = _module
test_fsdp_dsd = _module
test_compatibility = _module
test_dedup_tensors = _module
test_dtensor_checkpoint = _module
test_dtensor_resharding = _module
test_file_system_checkpoint = _module
test_file_system_checkpoint_cpu = _module
test_format_utils = _module
test_fsdp_model_state = _module
test_fsdp_optim_state = _module
test_fsdp_tp_checkpoint_conversion = _module
test_fsspec = _module
test_hsdp_checkpoint = _module
test_nested_dict = _module
test_planner = _module
test_save_load_api = _module
test_state_dict = _module
test_state_dict_utils = _module
test_tp_checkpoint = _module
test_traverse = _module
api_test = _module
local_elastic_agent_test = _module
lib_test = _module
metrics = _module
echo1 = _module
echo2 = _module
echo3 = _module
zombie_test = _module
error_handler_test = _module
redirects_test = _module
tail_log_test = _module
rendezvous = _module
c10d_rendezvous_backend_test = _module
dynamic_rendezvous_test = _module
etcd_rendezvous_backend_test = _module
etcd_rendezvous_test = _module
etcd_server_test = _module
out_of_tree_rendezvous_test = _module
testbackend = _module
rendezvous_backend_test = _module
static_rendezvous_test = _module
utils_test = _module
test_control_plane = _module
timer = _module
file_based_local_timer_test = _module
local_timer_example = _module
local_timer_test = _module
cycling_iterator_test = _module
distributed_test = _module
logging_test = _module
util_test = _module
test_fr_analysis = _module
test_checkpoint_wrapper = _module
test_distributed_checkpoint = _module
test_fsdp_apply = _module
test_fsdp_backward_prefetch = _module
test_fsdp_checkpoint = _module
test_fsdp_clip_grad_norm = _module
test_fsdp_comm = _module
test_fsdp_comm_hooks = _module
test_fsdp_core = _module
test_fsdp_dtensor_state_dict = _module
test_fsdp_exec_order = _module
test_fsdp_fine_tune = _module
test_fsdp_flatten_params = _module
test_fsdp_freezing_weights = _module
test_fsdp_fx = _module
test_fsdp_grad_acc = _module
test_fsdp_hybrid_shard = _module
test_fsdp_ignored_modules = _module
test_fsdp_input = _module
test_fsdp_memory = _module
test_fsdp_meta = _module
test_fsdp_misc = _module
test_fsdp_mixed_precision = _module
test_fsdp_multiple_forward = _module
test_fsdp_multiple_wrapping = _module
test_fsdp_overlap = _module
test_fsdp_pure_fp16 = _module
test_fsdp_sharded_grad_scaler = _module
test_fsdp_state_dict = _module
test_fsdp_tp_integration = _module
test_fsdp_traversal = _module
test_fsdp_uneven = _module
test_fsdp_unshard_params = _module
test_fsdp_use_orig_params = _module
test_hsdp_dtensor_state_dict = _module
test_shard_utils = _module
test_wrap = _module
test_script_init_method = _module
test_script_is_torchelastic_launched = _module
test_script_local_rank = _module
launch_test = _module
test_run = _module
test_instantiator = _module
test_apply_optimizer_in_backward = _module
test_named_optimizer = _module
test_zero_redundancy_optimizer = _module
pipelining = _module
model_registry = _module
schedule_registry = _module
test_backward = _module
test_microbatch = _module
test_pipe = _module
test_schedule = _module
test_schedule_multiproc = _module
test_stage = _module
test_transformer = _module
test_unflatten = _module
test_tensorpipe_agent = _module
test_faulty_agent = _module
test_share_memory = _module
parallel = _module
test_micro_pipeline_tp = _module
test_parallelize_api = _module
test_tp_examples = _module
test_tp_random_state = _module
test_tp_style = _module
test_c10d_common = _module
test_c10d_functional_native = _module
test_c10d_gloo = _module
test_c10d_logger = _module
test_c10d_nccl = _module
test_c10d_object_collectives = _module
test_c10d_ops_nccl = _module
test_c10d_pypg = _module
test_c10d_spawn = _module
test_c10d_spawn_gloo = _module
test_c10d_spawn_nccl = _module
test_c10d_spawn_ucc = _module
test_c10d_ucc = _module
test_collective_utils = _module
test_compute_comm_reordering = _module
test_control_collectives = _module
test_data_parallel = _module
test_device_mesh = _module
test_distributed_spawn = _module
test_dynamo_distributed = _module
test_fake_pg = _module
test_functional_api = _module
test_inductor_collectives = _module
test_launcher = _module
test_multi_threaded_pg = _module
test_nccl = _module
test_pg_wrapper = _module
test_store = _module
test_symmetric_memory = _module
test_constraints = _module
test_distributions = _module
test_transforms = _module
mock_modules = _module
mock_module1 = _module
mock_module2 = _module
mock_module3 = _module
mock_store_global_crossfile_inline = _module
test_activation_checkpointing = _module
test_after_aot = _module
test_aot_autograd = _module
test_aot_autograd_cache = _module
test_autograd_function = _module
test_backends = _module
test_backward_higher_order_ops = _module
test_base_output = _module
test_bytecode_utils = _module
test_compile = _module
test_compiler_bisector = _module
test_comptime = _module
test_config = _module
test_ctx_manager = _module
test_cudagraphs = _module
test_cudagraphs_expandable_segments = _module
test_debug_utils = _module
test_decorators = _module
test_deviceguard = _module
test_dynamic_shapes = _module
test_exc = _module
test_exceptions = _module
test_export = _module
test_export_mutations = _module
test_frame_init = _module
test_functions = _module
test_fx_passes_pre_grad = _module
test_global = _module
test_guard_manager = _module
test_higher_order_ops = _module
test_hooks = _module
test_inline_inbuilt_nn_modules = _module
test_input_attr_tracking = _module
test_interop = _module
test_minifier = _module
test_misc = _module
test_model_output = _module
test_modes = _module
test_modules = _module
test_nops = _module
test_pre_dispatch = _module
test_profiler = _module
test_python_autograd = _module
test_recompile_ux = _module
test_recompiles = _module
test_reconstruct = _module
test_reorder_logs = _module
test_repros = _module
test_resume = _module
test_sdpa = _module
test_skip_non_tensor = _module
test_sources = _module
test_structured_trace = _module
test_subclasses = _module
test_subgraphs = _module
test_torchrec = _module
test_trace_rules = _module
test_unspec = _module
test_verify_correctness = _module
test_view = _module
storage = _module
expect = _module
export = _module
opinfo_schema = _module
test_converter = _module
test_db = _module
test_experimental = _module
test_export_nonstrict = _module
test_export_training_ir_to_run_decomp = _module
test_functionalized_assertions = _module
test_hop = _module
test_lift_unlift = _module
test_pass_infra = _module
test_passes = _module
test_retraceability = _module
test_schema = _module
test_serdes = _module
test_serialize = _module
test_sparse = _module
test_swap = _module
test_tools = _module
test_torchbind = _module
test_tree_utils = _module
test_unflatten_training_ir = _module
test_verifier = _module
testing = _module
check_forward_backward_compatibility = _module
dump_all_function_schemas = _module
attn_ft = _module
attn_positional = _module
common_utils = _module
discover_coverage = _module
functorch_additional_op_db = _module
test_ac = _module
test_aotdispatch = _module
test_control_flow = _module
test_dims = _module
test_eager_transforms = _module
test_memory_efficient_fusion = _module
test_ops = _module
test_parsing = _module
test_rearrange = _module
test_vmap = _module
test_vmap_registrations = _module
xfail_suggester = _module
named_tup = _module
quantization = _module
test_common_passes = _module
test_cse_pass = _module
test_dce_pass = _module
test_future = _module
test_fx_const_fold = _module
test_fx_node_hook = _module
test_fx_param_shape_control_flow = _module
test_fx_split = _module
test_fx_xform_observer = _module
test_gradual_type = _module
test_lazy_graph_module = _module
test_matcher_utils = _module
test_partitioner_order = _module
test_shape_inference = _module
test_source_matcher_utils = _module
test_subgraph_rewriter = _module
test_z3_gradual_types = _module
hi = _module
test_invoke_subgraph = _module
test_with_effects = _module
inductor = _module
extension_codegen_backend = _module
device_interface = _module
extension_codegen_backend = _module
indirect_assert_helper = _module
minifier_smoke = _module
mock_cache = _module
opinfo_harness = _module
s429861_repro = _module
test_aot_inductor = _module
test_aot_inductor_arrayref = _module
test_aot_inductor_package = _module
test_aot_inductor_utils = _module
test_auto_functionalize = _module
test_autoheuristic = _module
test_b2b_gemm = _module
test_benchmark_fusion = _module
test_benchmarking = _module
test_binary_folding = _module
test_ck_backend = _module
test_codecache = _module
test_codegen_triton = _module
test_combo_kernels = _module
test_compile_worker = _module
test_compiled_autograd = _module
test_compiled_optimizers = _module
test_config = _module
test_cooperative_reductions = _module
test_coordinate_descent_tuner = _module
test_cpp_wrapper_hipify = _module
test_cpu_cpp_wrapper = _module
test_cpu_repro = _module
test_cpu_select_algorithm = _module
test_cuda_cpp_wrapper = _module
test_cuda_repro = _module
test_cudacodecache = _module
test_cudagraph_trees = _module
test_cudagraph_trees_expandable_segments = _module
test_custom_lowering = _module
test_custom_post_grad_passes = _module
test_cutlass_backend = _module
test_debug_trace = _module
test_decompose_mem_bound_mm = _module
test_dependencies = _module
test_distributed_patterns = _module
test_efficient_conv_bn_eval = _module
test_extension_backend = _module
test_external_callables = _module
test_flex_attention = _module
test_flex_decoding = _module
test_foreach = _module
test_fp8 = _module
test_fused_attention = _module
test_fx_fusion = _module
test_graph_transform_observer = _module
test_group_batch_fusion = _module
test_halide = _module
test_indexing = _module
test_inductor_freezing = _module
test_inductor_utils = _module
test_inplacing_pass = _module
test_kernel_benchmark = _module
test_layout_optim = _module
test_loop_ordering = _module
test_max_autotune = _module
test_memory = _module
test_memory_planning = _module
test_metrics = _module
test_minifier = _module
test_minifier_isolate = _module
test_mkldnn_pattern_matcher = _module
test_mmdecomp = _module
test_move_constructors_to_cuda = _module
test_multi_kernel = _module
test_ordered_set = _module
test_pad_mm = _module
test_padding = _module
test_pattern_matcher = _module
test_perf = _module
test_profiler = _module
test_scatter_optimization = _module
test_select_algorithm = _module
test_smoke = _module
test_snode_runtime = _module
test_split_cat_fx_passes = _module
test_standalone_compile = _module
test_torchinductor = _module
test_torchinductor_codegen_dynamic_shapes = _module
test_torchinductor_dynamic_shapes = _module
test_torchinductor_opinfo = _module
test_torchinductor_strided_blocks = _module
test_triton_cpu_backend = _module
test_triton_extension_backend = _module
test_triton_heuristics = _module
test_triton_kernels = _module
test_triton_wrapper = _module
test_unbacked_symints = _module
test_xpu_basic = _module
_imported_class_test = _module
bar = _module
foo = _module
very = _module
nested = _module
fixtures_srcs = _module
fixtures_src = _module
generate_models = _module
test_upgrader_models_generation = _module
mydecorator = _module
myexception = _module
myfunction_a = _module
myfunction_b = _module
test_alias_analysis = _module
test_async = _module
test_aten_pow = _module
test_attr = _module
test_autodiff = _module
test_autodiff_subgraph_slicing = _module
test_await = _module
test_backend_nnapi = _module
test_batch_mm = _module
test_builtins = _module
test_class_type = _module
test_complexity = _module
test_convert_activation = _module
test_custom_operators = _module
test_dataclasses = _module
test_dce = _module
test_decorator = _module
test_device_analysis = _module
test_dtype_analysis = _module
test_enum = _module
test_exception = _module
test_freezing = _module
test_functional_blocks = _module
test_fuser_common = _module
test_generator = _module
test_graph_rewrite_passes = _module
test_hash = _module
test_hooks_modules = _module
test_ignorable_args = _module
test_ignore_context_manager = _module
test_isinstance = _module
test_jit_utils = _module
test_list_dict = _module
test_models = _module
test_module_apis = _module
test_module_containers = _module
test_module_interface = _module
test_op_decompositions = _module
test_optimize_for_mobile_preserve_debug_info = _module
test_pdt = _module
test_peephole = _module
test_python_bindings = _module
test_python_builtins = _module
test_python_ir = _module
test_recursive_script = _module
test_remove_mutation = _module
test_save_load = _module
test_save_load_for_op_version = _module
test_script_profile = _module
test_scriptmod_ann = _module
test_slice = _module
test_string_formatting = _module
test_symbolic_shape_analysis = _module
test_tensor_creation_ops = _module
test_tensor_methods = _module
test_tracer = _module
test_type_sharing = _module
test_types = _module
test_typing = _module
test_union = _module
test_union_pep604 = _module
test_unsupported_ops = _module
test_upgraders = _module
test_warn = _module
test_with = _module
test_xnnpack_delegate = _module
lazy = _module
test_bindings = _module
test_debug_util = _module
test_extract_compiled_graph = _module
test_functionalization = _module
test_meta_kernel = _module
test_reuse_ir = _module
test_step_closures = _module
test_ts_opinfo = _module
linear = _module
load_torchscript_model = _module
mkl_verbose = _module
mkldnn_verbose = _module
prepare_model = _module
android_api_module = _module
builtin_ops = _module
gen_test_model = _module
math_ops = _module
nn_ops = _module
quantization_ops = _module
sampling_ops = _module
tensor_ops = _module
update_production_ops = _module
aot_test_model = _module
test_bytecode = _module
test_lite_script_module = _module
test_lite_script_type = _module
test_quantize_fx_lite_script_module = _module
test_upgrader_codegen = _module
test_convolution = _module
test_dropout = _module
test_lazy_modules = _module
test_load_state_dict = _module
test_module_hooks = _module
test_multihead_attention = _module
test_packed_sequence = _module
test_pooling = _module
test_pruning = _module
autograd_helper = _module
test_dynamo_with_onnxruntime_backend = _module
test_exporter_api = _module
test_registry_dispatcher = _module
error_reproduction = _module
test_capture_strategies = _module
test_core = _module
test_small_models_e2e = _module
test_tensors = _module
test_diagnostics = _module
test_registraion = _module
model_defs = _module
dcgan = _module
emb_seq = _module
lstm_flattening_result = _module
mnist = _module
op_test = _module
rnn_model_with_packed_sequence = _module
squeezenet = _module
srresnet = _module
super_resolution = _module
word_language_model = _module
onnx_test_common = _module
pytorch_test_common = _module
test_autograd_funs = _module
test_fx_passes = _module
test_fx_to_onnx = _module
test_fx_to_onnx_decomp_skip = _module
test_fx_type_promotion = _module
test_lazy_import = _module
test_models_onnxruntime = _module
test_models_quantized_onnxruntime = _module
test_onnx_opset = _module
test_onnxscript_no_runtime = _module
test_onnxscript_runtime = _module
test_op_consistency = _module
test_pytorch_jit_onnx = _module
test_pytorch_onnx_no_runtime = _module
test_pytorch_onnx_onnxruntime = _module
test_pytorch_onnx_onnxruntime_cuda = _module
test_pytorch_onnx_shape_inference = _module
test_symbolic_helper = _module
test_utility_funs = _module
test_verification = _module
test_torch_export_with_onnxruntime = _module
verify = _module
test_lrscheduler = _module
test_optim = _module
test_swa_utils = _module
package = _module
generate_bc_packages = _module
module_a = _module
module_a_remapped_path = _module
package_a = _module
fake_interface = _module
fake_script_class = _module
long_name = _module
std_sys_module_hacks = _module
subpackage = _module
test_all_leaf_modules_tracer = _module
test_module = _module
test_nn_module = _module
use_dunder_package = _module
use_torch_package_importer = _module
package_b = _module
subpackage_0 = _module
subsubpackage_0 = _module
subpackage_1 = _module
subpackage_2 = _module
package_c = _module
package_d = _module
imports_directly = _module
imports_indirectly = _module
test_analyze = _module
test_dependency_api = _module
test_dependency_hooks = _module
test_digraph = _module
test_directory_reader = _module
test_glob_group = _module
test_importer = _module
test_load_bc_packages = _module
test_mangling = _module
test_model = _module
test_package_fx = _module
test_package_script = _module
test_repackage = _module
test_resources = _module
test_trace_dep = _module
test_cpp_thread = _module
test_execution_trace = _module
test_memory_profiler = _module
test_profiler_tree = _module
test_record_function = _module
test_torch_tidy = _module
pytest_shard_custom = _module
ao_migration = _module
test_ao_migration = _module
test_quantization_fx = _module
bc = _module
test_backward_compatibility = _module
apot_fx_graph_mode_ptq = _module
apot_fx_graph_mode_qat = _module
quantization_util = _module
test_adaround_eager = _module
test_bits = _module
test_fake_quantize = _module
test_float8 = _module
test_linear = _module
test_nonuniform_observer = _module
test_quantized_tensor = _module
test_quantizer = _module
test_backend_config = _module
test_docs = _module
test_quantized_functional = _module
test_quantized_module = _module
test_quantized_op = _module
test_top_level_apis = _module
test_workflow_module = _module
test_workflow_ops = _module
eager = _module
test_bias_correction_eager = _module
test_equalize_eager = _module
test_fuse_eager = _module
test_model_numerics = _module
test_numeric_suite_eager = _module
test_quantize_eager_ptq = _module
test_quantize_eager_qat = _module
fx = _module
test_equalize_fx = _module
test_model_report_fx = _module
test_numeric_suite_fx = _module
test_quantize_fx = _module
test_deprecated_jit_quant = _module
test_fusion_passes = _module
test_ondevice_quantization = _module
test_quantize_jit = _module
test_duplicate_dq = _module
test_graph_utils = _module
test_metadata_porting = _module
test_numeric_debugger = _module
test_quantize_pt2e = _module
test_quantize_pt2e_qat = _module
test_representation = _module
test_x86inductor_quantizer = _module
test_xnnpack_quantizer = _module
run_test = _module
cuda_memcheck_common = _module
run_cuda_memcheck = _module
simulate_nccl_errors = _module
test_ao_sparsity = _module
test_autocast = _module
test_autograd = _module
test_autograd_fallback = _module
test_autoload = _module
test_binary_ufuncs = _module
test_bundled_images = _module
test_bundled_inputs = _module
test_ci_sanity_check_fail = _module
test_comparison_utils = _module
test_compile_benchmark_util = _module
test_content_store = _module
test_cpp_api_parity = _module
test_cpp_extensions_aot = _module
test_cpp_extensions_jit = _module
test_cpp_extensions_mtia_backend = _module
test_cpp_extensions_open_device_registration = _module
test_cpp_extensions_stream_and_event = _module
test_cuda = _module
test_cuda_expandable_segments = _module
test_cuda_multigpu = _module
test_cuda_nvml_based_avail = _module
test_cuda_primary_ctx = _module
test_cuda_sanitizer = _module
test_cuda_trace = _module
test_dataloader = _module
test_datapipe = _module
test_decomp = _module
test_deploy = _module
test_determination = _module
test_dispatch = _module
test_dlpack = _module
test_expanded_weights = _module
test_fake_tensor = _module
test_file_check = _module
test_flop_counter = _module
test_function_schema = _module
test_functional_autograd_benchmark = _module
test_functional_optim = _module
test_functionalization_of_rng_ops = _module
test_futures = _module
test_fx = _module
test_fx_experimental = _module
test_fx_reinplace_pass = _module
test_hub = _module
test_import_stats = _module
test_itt = _module
test_jit = _module
test_jit_autocast = _module
test_jit_disabled = _module
test_jit_fuser = _module
test_jit_fuser_legacy = _module
test_jit_fuser_te = _module
test_jit_legacy = _module
test_jit_llga_fuser = _module
test_jit_profiling = _module
test_jit_simple = _module
test_jit_string = _module
test_jiterator = _module
test_kernel_launch_checks = _module
test_legacy_vmap = _module
test_license = _module
test_linalg = _module
test_masked = _module
test_maskedtensor = _module
test_matmul_cuda = _module
test_meta = _module
test_metal = _module
test_mkl_verbose = _module
test_mkldnn = _module
test_mkldnn_fusion = _module
test_mkldnn_verbose = _module
test_mobile_optimizer = _module
test_model_dump = _module
test_model_exports_to_core_aten = _module
test_module_tracker = _module
test_monitor = _module
test_mps = _module
test_multiprocessing = _module
test_multiprocessing_spawn = _module
test_namedtensor = _module
test_namedtuple_return_api = _module
test_native_functions = _module
test_native_mha = _module
test_nestedtensor = _module
test_nn = _module
test_nnapi = _module
test_numba_integration = _module
test_numpy_interop = _module
test_openmp = _module
test_ops_fwd_gradients = _module
test_ops_gradients = _module
test_ops_jit = _module
test_out_dtype_op = _module
test_overrides = _module
test_package = _module
test_per_overload_api = _module
test_prims = _module
test_proxy_tensor = _module
test_pruning_op = _module
test_public_bindings = _module
test_python_dispatch = _module
test_pytree = _module
test_reductions = _module
test_scatter_gather_ops = _module
test_schema_check = _module
test_segment_reductions = _module
test_serialization = _module
test_set_default_mobile_cpu_allocator = _module
test_shape_ops = _module
test_show_pickle = _module
test_sort_and_select = _module
test_sparse_csr = _module
test_sparse_semi_structured = _module
test_spectral_ops = _module
test_stateless = _module
test_static_runtime = _module
test_subclass = _module
test_sympy_utils = _module
test_tensorboard = _module
test_tensorexpr = _module
test_tensorexpr_pybind = _module
test_testing = _module
test_throughput_benchmark = _module
test_torch = _module
test_transformers = _module
test_type_hints = _module
test_type_info = _module
test_type_promotion = _module
test_unary_ufuncs = _module
test_utils_config_module = _module
test_utils_internal = _module
test_vulkan = _module
test_weak = _module
test_xnnpack_integration = _module
test_xpu = _module
torch_np = _module
check_tests_conform = _module
test_dtype = _module
test_einsum = _module
test_getlimits = _module
test_multiarray = _module
test_numeric = _module
test_numerictypes = _module
test_scalar_ctors = _module
test_scalar_methods = _module
test_scalarinherit = _module
test_scalarmath = _module
test_shape_base = _module
test_helper = _module
test_pocketfft = _module
test_arraypad = _module
test_arraysetops = _module
test_function_base = _module
test_histograms = _module
test_index_tricks = _module
test_shape_base_ = _module
test_twodim_base = _module
test_type_check = _module
test_basic = _module
test_ndarray_methods = _module
test_nep50_examples = _module
test_random = _module
test_scalars_0D_arrays = _module
test_ufuncs_basic = _module
creation_ops = _module
disabled_bitwise_ops = _module
random = _module
cuda_steam = _module
disabled_jit = _module
module_list = _module
namedtuple = _module
opt_size = _module
size = _module
tensor_constructors = _module
tensor_copy = _module
tensor_sampling = _module
torch_optim = _module
test_conv = _module
test_gemm = _module
build_bundled = _module
tools = _module
alerts = _module
create_alerts = _module
build_amd = _module
autograd = _module
context = _module
gen_annotated_fn_args = _module
gen_autograd = _module
gen_autograd_functions = _module
gen_inplace_or_view_type = _module
gen_python_functions = _module
gen_trace_type = _module
gen_variable_factories = _module
gen_variable_type = _module
gen_view_funcs = _module
load_derivatives = _module
build_libtorch = _module
build_pytorch_libs = _module
build_with_debinfo = _module
gen_op_registration_allowlist = _module
gen_operators_yaml = _module
gen_oplist = _module
oss_coverage = _module
oss = _module
cov_json = _module
init = _module
run = _module
tool = _module
clang_coverage = _module
gcc_coverage = _module
parser = _module
coverage_record = _module
gcov_coverage_parser = _module
llvm_coverage_parser = _module
llvm_coverage_segment = _module
print_report = _module
summarize_jsons = _module
util = _module
setting = _module
utils_init = _module
coverage_plugins = _module
jit_plugin = _module
download_mnist = _module
verify_dynamo = _module
extract_scripts = _module
builder = _module
config_manager = _module
loader = _module
fr_trace = _module
gen_vulkan_spv = _module
generate_torch_version = _module
github = _module
fixup = _module
gen_unboxing = _module
test_gen_unboxing = _module
linter = _module
actionlint_linter = _module
bazel_linter = _module
black_linter = _module
clangformat_linter = _module
clangtidy_linter = _module
cmake_linter = _module
exec_linter = _module
flake8_linter = _module
grep_linter = _module
lintrunner_version_linter = _module
mypy_linter = _module
nativefunctions_linter = _module
newlines_linter = _module
no_merge_conflict_csv_linter = _module
pip_init = _module
pyfmt_linter = _module
ruff_linter = _module
s3_init = _module
shellcheck_linter = _module
test_has_main_linter = _module
testowners_linter = _module
update_s3 = _module
workflow_consistency_linter = _module
clang_tidy = _module
generate_build_files = _module
lite_interpreter = _module
gen_selected_mobile_ops_header = _module
deploy_debugger = _module
pytorch_lldb = _module
nightly = _module
nightly_hotpatch = _module
nvcc_fix_deps = _module
gen_diagnostics = _module
update_default_opset_version = _module
pyi = _module
gen_pyi = _module
render_junit = _module
setup_helpers = _module
cmake = _module
cmake_utils = _module
env = _module
gen = _module
gen_version_header = _module
generate_code = _module
generate_linker_script = _module
shared = _module
logging_utils = _module
module_loader = _module
stats = _module
check_disabled_tests = _module
export_test_times = _module
import_test_stats = _module
monitor = _module
test_dashboard = _module
upload_artifacts = _module
upload_dynamo_perf_stats = _module
upload_external_contrib_stats = _module
upload_metrics = _module
upload_sccache_stats = _module
upload_stats_lib = _module
upload_test_stat_aggregates = _module
upload_test_stats = _module
upload_test_stats_intermediate = _module
substitute = _module
gen_operators_yaml_test = _module
gen_oplist_test = _module
heuristics = _module
test_heuristics = _module
test_interface = _module
test_cmake = _module
test_codegen = _module
test_codegen_model = _module
test_create_alerts = _module
test_executorch_custom_ops = _module
test_executorch_gen = _module
test_executorch_signatures = _module
test_executorch_types = _module
test_executorch_unboxing = _module
test_gen_backend_stubs = _module
test_selective_build = _module
test_test_run = _module
test_test_selections = _module
test_upload_stats_lib = _module
test_upload_test_stats = _module
test_vulkan_codegen = _module
discover_tests = _module
do_target_determination_for_s3 = _module
explicit_ci_jobs = _module
modulefinder_determinator = _module
determinator = _module
gen_artifact = _module
correlated_with_historical_failures = _module
edited_by_pr = _module
filepath = _module
historical_class_failure_correlation = _module
historical_edited_files = _module
interface = _module
llm = _module
mentioned_in_pr = _module
previously_failed_in_pr = _module
profiling = _module
public_bindings = _module
test_selections = _module
update_slow_tests = _module
update_masked_docs = _module
vscode_settings = _module
_VF = _module
__config__ = _module
__future__ = _module
_appdirs = _module
_awaits = _module
_classes = _module
_compile = _module
_custom_op = _module
functional = _module
impl = _module
_custom_ops = _module
_decomp = _module
decompositions = _module
decompositions_for_jvp = _module
decompositions_for_rng = _module
_deploy = _module
_dispatch = _module
python = _module
_dynamo = _module
_trace_wrapped_higher_order_op = _module
backends = _module
cudagraphs = _module
debugging = _module
onnxrt = _module
registry = _module
tensorrt = _module
torchxla = _module
tvm = _module
bytecode_analysis = _module
bytecode_transformation = _module
cache_size = _module
callback = _module
code_context = _module
compiled_autograd = _module
comptime = _module
config = _module
convert_frame = _module
create_parameter_op = _module
current_scope_id = _module
debug_utils = _module
decorators = _module
eval_frame = _module
exc = _module
external_utils = _module
funcname_cache = _module
guards = _module
logging = _module
mutation_guard = _module
output_graph = _module
pgo = _module
polyfills = _module
builtins = _module
functools = _module
itertools = _module
os = _module
sys = _module
profiler = _module
replay_record = _module
repro = _module
after_aot = _module
after_dynamo = _module
resume_execution = _module
side_effects = _module
source = _module
symbolic_convert = _module
tensor_version_op = _module
test_case = _module
test_minifier_common = _module
trace_rules = _module
utils = _module
variables = _module
base = _module
builder = _module
builtin = _module
constant = _module
ctx_manager = _module
dicts = _module
functions = _module
higher_order_ops = _module
iter = _module
lists = _module
misc = _module
nn_module = _module
optimizer = _module
script_object = _module
tensor = _module
torch_function = _module
user_defined = _module
_environment = _module
_export = _module
converter = _module
db = _module
case = _module
examples = _module
assume_constant_result = _module
autograd_function = _module
class_method = _module
cond_branch_class_method = _module
cond_branch_nested_function = _module
cond_branch_nonlocal_variables = _module
cond_closed_over_variable = _module
cond_operands = _module
cond_predicate = _module
constrain_as_size_example = _module
constrain_as_value_example = _module
decorator = _module
dictionary = _module
dynamic_shape_assert = _module
dynamic_shape_constructor = _module
dynamic_shape_if_guard = _module
dynamic_shape_map = _module
dynamic_shape_round = _module
dynamic_shape_slicing = _module
dynamic_shape_view = _module
fn_with_kwargs = _module
list_contains = _module
list_unpack = _module
model_attr_mutation = _module
nested_function = _module
null_context_manager = _module
optional_input = _module
pytree_flatten = _module
scalar_output = _module
specialized_attribute = _module
static_for_loop = _module
static_if = _module
tensor_setattr = _module
type_reflection_method = _module
unsupported_operator = _module
user_input_mutation = _module
gen_example = _module
error = _module
non_strict_utils = _module
pass_base = _module
pass_infra = _module
node_metadata = _module
proxy_value = _module
passes = _module
_node_metadata_hook = _module
add_runtime_assertions_for_constraints_pass = _module
collect_tracepoints_pass = _module
constant_folding = _module
functionalize_side_effectful_ops_pass = _module
lift_constants_pass = _module
remove_runtime_assertions = _module
replace_autocast_with_hop_pass = _module
replace_quantized_ops_with_standard_ops_pass = _module
replace_set_grad_with_hop_pass = _module
replace_view_ops_with_view_copy_ops_pass = _module
replace_with_hop_pass_util = _module
serde = _module
aoti_schema = _module
dynamic_shapes = _module
schema = _module
schema_check = _module
serialize = _module
union = _module
verifier = _module
wrappers = _module
_functorch = _module
_aot_autograd = _module
autograd_cache = _module
collect_metadata_analysis = _module
dispatch_and_compile_graph = _module
functional_utils = _module
input_output_analysis = _module
jit_compile_runtime_wrappers = _module
runtime_wrappers = _module
schemas = _module
subclass_utils = _module
traced_function_transforms = _module
aot_autograd = _module
apis = _module
batch_norm_replacement = _module
compile_utils = _module
compilers = _module
deprecated = _module
functional_call = _module
fx_minifier = _module
partitioners = _module
pyfunctorch = _module
python_key = _module
pytree_hacks = _module
top_operators_github_usage = _module
_guards = _module
_higher_order_ops = _module
associative_scan = _module
auto_functionalize = _module
cond = _module
effects = _module
executorch_call_delegate = _module
flex_attention = _module
hints_wrap = _module
invoke_subgraph = _module
map = _module
out_dtype = _module
run_const_graph = _module
scan = _module
strict_mode = _module
torchbind = _module
triton_kernel_wrap = _module
while_loop = _module
wrap = _module
_inductor = _module
aoti_eager = _module
async_compile = _module
autoheuristic = _module
_MMRankingA100 = _module
_MMRankingH100 = _module
_MixedMMA100 = _module
_MixedMMH100 = _module
_PadMMA100 = _module
artifacts = _module
autoheuristic_utils = _module
learned_heuristic_controller = _module
learnedheuristic_interface = _module
autotune_process = _module
bisect_helper = _module
bounds = _module
codecache = _module
aoti_hipify_utils = _module
common = _module
cpp = _module
cpp_gemm_template = _module
cpp_micro_gemm = _module
cpp_template = _module
cpp_template_kernel = _module
cpp_utils = _module
cpp_wrapper_cpu = _module
cpp_wrapper_cpu_array_ref = _module
cpp_wrapper_gpu = _module
cpu_device_op_overrides = _module
cuda = _module
cuda_cpp_scheduling = _module
cuda_env = _module
cuda_kernel = _module
cuda_template = _module
cutlass_epilogue_gen = _module
cutlass_lib_extensions = _module
gemm_operation_extensions = _module
cutlass_utils = _module
device_op_overrides = _module
gemm_template = _module
cuda_combined_scheduling = _module
halide = _module
memory_planning = _module
multi_kernel = _module
rocm = _module
ck_conv_template = _module
ck_template = _module
ck_universal_gemm_template = _module
compile_command = _module
rocm_benchmark_request = _module
rocm_cpp_scheduling = _module
rocm_kernel = _module
rocm_template = _module
rocm_template_buffer = _module
simd = _module
triton = _module
triton_combo_kernel = _module
triton_split_scan = _module
triton_utils = _module
wrapper = _module
xpu = _module
comm_analysis = _module
comms = _module
compile_fx = _module
compile_worker = _module
subproc_pool = _module
watchdog = _module
config = _module
constant_folding = _module
cpp_builder = _module
cpu_vec_isa = _module
cudagraph_trees = _module
cudagraph_utils = _module
custom_graph_pass = _module
debug = _module
decomposition = _module
dependencies = _module
extern_node_serializer = _module
freezing = _module
fx_passes = _module
b2b_gemm = _module
binary_folding = _module
ddp_fusion = _module
decompose_mem_bound_mm = _module
dedupe_symint_uses = _module
efficient_conv_bn_eval = _module
freezing_patterns = _module
fuse_attention = _module
group_batch_fusion = _module
joint_graph = _module
micro_pipeline_tp = _module
misc_patterns = _module
mkldnn_fusion = _module
numeric_utils = _module
pad_mm = _module
post_grad = _module
pre_grad = _module
reinplace = _module
replace_random = _module
serialized_patterns = _module
_sfdp_pattern_1 = _module
_sfdp_pattern_10 = _module
_sfdp_pattern_11 = _module
_sfdp_pattern_12 = _module
_sfdp_pattern_13 = _module
_sfdp_pattern_14 = _module
_sfdp_pattern_15 = _module
_sfdp_pattern_16 = _module
_sfdp_pattern_17 = _module
_sfdp_pattern_18 = _module
_sfdp_pattern_19 = _module
_sfdp_pattern_2 = _module
_sfdp_pattern_3 = _module
_sfdp_pattern_4 = _module
_sfdp_pattern_5 = _module
_sfdp_pattern_6 = _module
_sfdp_pattern_7 = _module
_sfdp_pattern_8 = _module
_sfdp_pattern_9 = _module
addmm_pattern = _module
bmm_pattern = _module
mm_pattern = _module
split_cat = _module
fx_utils = _module
graph = _module
index_propagation = _module
inductor_prims = _module
ir = _module
jagged_lowerings = _module
kernel = _module
bmm = _module
conv = _module
flex_attention = _module
flex_decoding = _module
mm = _module
mm_common = _module
mm_plus_mm = _module
mm_scaled = _module
unpack_mixed_mm = _module
loop_body = _module
lowering = _module
memory = _module
metrics = _module
mkldnn_ir = _module
mkldnn_lowerings = _module
ops_handler = _module
optimize_indexing = _module
build_package = _module
pt2_archive_constants = _module
pattern_matcher = _module
quantized_lowerings = _module
remote_cache = _module
runtime = _module
autotune_cache = _module
benchmarking = _module
cache_dir_utils = _module
compile_tasks = _module
coordinate_descent_tuner = _module
halide_helpers = _module
hints = _module
runtime_utils = _module
triton_helpers = _module
triton_heuristics = _module
scheduler = _module
select_algorithm = _module
sizevars = _module
subgraph_lowering = _module
test_operators = _module
utils = _module
virtualized = _module
wrapper_benchmark = _module
_jit_internal = _module
_lazy = _module
closure = _module
computation = _module
device_context = _module
extract_compiled_graph = _module
ir_cache = _module
tensor_factory_functions = _module
ts_backend = _module
_library = _module
custom_ops = _module
fake_class_registry = _module
fake_impl = _module
infer_schema = _module
simple_registry = _module
triton = _module
_linalg_utils = _module
_lobpcg = _module
_logging = _module
_internal = _module
_registrations = _module
scribe = _module
structured = _module
_lowrank = _module
_meta_registrations = _module
_namedtensor_internals = _module
_numpy = _module
_binary_ufuncs_impl = _module
_casting_dicts = _module
_dtypes = _module
_dtypes_impl = _module
_funcs = _module
_funcs_impl = _module
_getlimits = _module
_ndarray = _module
_normalizations = _module
_reductions_impl = _module
_ufuncs = _module
_unary_ufuncs_impl = _module
_util = _module
fft = _module
linalg = _module
_ops = _module
_prims = _module
debug_prims = _module
executor = _module
rng_prims = _module
_prims_common = _module
_python_dispatcher = _module
_refs = _module
_conversions = _module
nn = _module
special = _module
_size_docs = _module
_sources = _module
_storage_docs = _module
_streambase = _module
_strobelight = _module
cli_function_profiler = _module
compile_time_profiler = _module
cli_function_profiler_example = _module
compile_time_profile_example = _module
_subclasses = _module
_fake_tensor_utils = _module
fake_impls = _module
fake_tensor = _module
fake_utils = _module
functional_tensor = _module
meta_utils = _module
schema_check_mode = _module
_tensor_docs = _module
_tensor_str = _module
_thread_safe_fork = _module
_torch_docs = _module
_utils = _module
_utils_internal = _module
_vendor = _module
packaging = _module
_structures = _module
version = _module
_vmap_internals = _module
_weights_only_unpickler = _module
accelerator = _module
amp = _module
autocast_mode = _module
grad_scaler = _module
ao = _module
intrinsic = _module
modules = _module
fused = _module
qat = _module
conv_fused = _module
linear_fused = _module
linear_relu = _module
quantized = _module
dynamic = _module
bn_relu = _module
conv_add = _module
conv_relu = _module
embedding_ops = _module
quantizable = _module
activation = _module
rnn = _module
batchnorm = _module
dropout = _module
functional_modules = _module
ns = _module
_numeric_suite = _module
_numeric_suite_fx = _module
graph_matcher = _module
graph_passes = _module
mappings = _module
n_shadows_utils = _module
ns_types = _module
pattern_utils = _module
qconfig_multi_mapping = _module
weight_utils = _module
pruning = _module
_experimental = _module
activation_sparsifier = _module
data_scheduler = _module
base_data_scheduler = _module
data_sparsifier = _module
base_data_sparsifier = _module
dlrm_utils = _module
evaluate_disk_savings = _module
evaluate_forward_time = _module
evaluate_model_metrics = _module
data_norm_sparsifier = _module
lightning = _module
callbacks = _module
_data_sparstity_utils = _module
data_sparsity = _module
test_callbacks = _module
quantization_utils = _module
FPGM_pruner = _module
pruner = _module
base_structured_sparsifier = _module
lstm_saliency_pruner = _module
match_utils = _module
parametrization = _module
prune_functions = _module
saliency_pruner = _module
_mappings = _module
base_scheduler = _module
cubic_scheduler = _module
lambda_scheduler = _module
sparsifier = _module
base_sparsifier = _module
nearly_diagonal_sparsifier = _module
weight_norm_sparsifier = _module
_correct_bias = _module
_equalize = _module
_learnable_fake_quantize = _module
backend_config = _module
_common_operator_config_utils = _module
_qnnpack_pt2e = _module
executorch = _module
fbgemm = _module
native = _module
observation_type = _module
onednn = _module
qnnpack = _module
x86 = _module
APoT_tensor = _module
adaround_fake_quantize = _module
adaround_loss = _module
adaround_optimization = _module
apot_utils = _module
fake_quantize = _module
fake_quantize_function = _module
qconfig = _module
quantizer = _module
fuse_modules = _module
fuser_method_mappings = _module
_decomposed = _module
_lower_to_native_backend = _module
_model_report = _module
detector = _module
model_report = _module
model_report_observer = _module
model_report_visualizer = _module
convert = _module
custom_config = _module
fuse = _module
fuse_handler = _module
graph_module = _module
lower_to_fbgemm = _module
lower_to_qnnpack = _module
lstm_utils = _module
prepare = _module
qconfig_mapping_utils = _module
quantize_handler = _module
tracer = _module
pt2e = _module
_numeric_debugger = _module
duplicate_dq_pass = _module
export_utils = _module
graph_utils = _module
port_metadata_pass = _module
qat_utils = _module
representation = _module
rewrite = _module
qconfig_mapping = _module
quant_type = _module
quantization_mappings = _module
quantize_fx = _module
quantize_jit = _module
quantize_pt2e = _module
composable_quantizer = _module
embedding_quantizer = _module
x86_inductor_quantizer = _module
xnnpack_quantizer = _module
xnnpack_quantizer_utils = _module
stubs = _module
_functions = _module
anomaly_mode = _module
forward_ad = _module
function = _module
grad_mode = _module
gradcheck = _module
profiler_legacy = _module
profiler_util = _module
variable = _module
_coreml = _module
preprocess = _module
_nnapi = _module
serializer = _module
cpu = _module
cudnn = _module
cusparselt = _module
mha = _module
mkl = _module
mkldnn = _module
mps = _module
nnpack = _module
openmp = _module
opt_einsum = _module
xeon = _module
run_cpu = _module
xnnpack = _module
compiler = _module
contrib = _module
_tensorboard_vis = _module
codegen_external = _module
bisect = _module
test_mnist = _module
cuda = _module
_gpu_trace = _module
_memory_viz = _module
_sanitizer = _module
comm = _module
gds = _module
graphs = _module
jiterator = _module
nccl = _module
nvtx = _module
streams = _module
tunable = _module
_checkpointable = _module
_composable = _module
checkpoint_activation = _module
contract = _module
fsdp = _module
_fsdp_api = _module
_fsdp_collectives = _module
_fsdp_common = _module
_fsdp_init = _module
_fsdp_param = _module
_fsdp_param_group = _module
_fsdp_state = _module
fully_shard = _module
replicate = _module
_composable_state = _module
_functional_collectives = _module
_functional_collectives_impl = _module
_shard = _module
checkpoint = _module
common_op_utils = _module
metadata = _module
op_registry_utils = _module
sharded_optim = _module
sharded_tensor = _module
_common = _module
binary_cmp = _module
misc_ops = _module
logger = _module
logging_handlers = _module
reshard = _module
shard = _module
sharder = _module
sharding_plan = _module
sharding_spec = _module
_internals = _module
chunk_sharding_spec = _module
chunk_sharding_spec_ops = _module
embedding = _module
embedding_bag = _module
_sharded_tensor = _module
_sharding_spec = _module
_state_dict_utils = _module
_symmetric_memory = _module
placement_types = _module
_tools = _module
fsdp2_mem_tracker = _module
ilp_utils = _module
mem_tracker = _module
memory_tracker = _module
mod_tracker = _module
runtime_estimator = _module
sac_estimator = _module
sac_ilp = _module
algorithms = _module
_checkpoint = _module
checkpoint_wrapper = _module
_comm_hooks = _module
default_hooks = _module
_optimizer_overlap = _module
optimizer_overlap = _module
_quantization = _module
ddp_comm_hooks = _module
ddp_zero_hook = _module
debugging_hooks = _module
mixed_precision_hooks = _module
optimizer_overlap_hooks = _module
post_localSGD_hook = _module
powerSGD_hook = _module
quantization_hooks = _module
join = _module
model_averaging = _module
averagers = _module
hierarchical_model_averager = _module
argparse_util = _module
benchmark_ddp_rpc = _module
c10d_logger = _module
_checkpointer = _module
_dedup_save_plans = _module
_dedup_tensors = _module
_fsspec_filesystem = _module
_nested_dict = _module
_sharded_tensor_utils = _module
_storage_utils = _module
_traverse = _module
_version = _module
default_planner = _module
async_checkpointing_example = _module
fsdp_checkpoint_example = _module
stateful_example = _module
filesystem = _module
format_utils = _module
planner = _module
planner_helpers = _module
resharding = _module
staging = _module
state_dict = _module
state_dict_loader = _module
state_dict_saver = _module
stateful = _module
collective_utils = _module
constants = _module
device_mesh = _module
distributed_c10d = _module
elastic = _module
health_check_server = _module
local_elastic_agent = _module
control_plane = _module
events = _module
handlers = _module
multiprocessing = _module
errors = _module
error_handler = _module
redirects = _module
subprocess_handler = _module
tail_log = _module
c10d_rendezvous_backend = _module
dynamic_rendezvous = _module
etcd_rendezvous = _module
etcd_rendezvous_backend = _module
etcd_server = _module
etcd_store = _module
static_tcp_rendezvous = _module
debug_info_logging = _module
file_based_local_timer = _module
local_timer = _module
cycling_iterator = _module
elastic_distributed_sampler = _module
log_level = _module
store = _module
memory_tracker_example = _module
_common_utils = _module
_debug_utils = _module
_dynamo_utils = _module
_exec_order_utils = _module
_flat_param = _module
_fsdp_extensions = _module
_init_utils = _module
_limiter_utils = _module
_optim_utils = _module
_runtime_utils = _module
_shard_utils = _module
_trace_utils = _module
_traversal_utils = _module
_unshard_param_utils = _module
_wrap_utils = _module
fully_sharded_data_parallel = _module
sharded_grad_scaler = _module
launch = _module
remote_module = _module
instantiator = _module
templates = _module
remote_module_template = _module
optim = _module
apply_optimizer_in_backward = _module
functional_adadelta = _module
functional_adagrad = _module
functional_adam = _module
functional_adamax = _module
functional_adamw = _module
functional_rmsprop = _module
functional_rprop = _module
functional_sgd = _module
named_optimizer = _module
post_localSGD_optimizer = _module
zero_redundancy_optimizer = _module
_IR = _module
_backward = _module
_debug = _module
_unflatten = _module
microbatch = _module
schedules = _module
stage = _module
remote_device = _module
rpc = _module
_testing = _module
faulty_agent_backend_registry = _module
backend_registry = _module
internal = _module
options = _module
rref_proxy = _module
server_process_global_profiler = _module
_api = _module
_collective_utils = _module
_dtensor_spec = _module
_op_schema = _module
_common_rules = _module
_conv_ops = _module
_einsum_strategy = _module
_embedding_ops = _module
_experimental_ops = _module
_math_ops = _module
_matrix_ops = _module
_pointwise_ops = _module
_random_ops = _module
_tensor_ops = _module
_view_ops = _module
_random = _module
_redistribute = _module
_sharding_prop = _module
_shards_wrapper = _module
_tp_conv = _module
_comm_mode = _module
_op_coverage = _module
_visualize_sharding = _module
comm_mode_features_example = _module
convnext_example = _module
torchrec_sharding_example = _module
visualize_sharding_example = _module
_attention = _module
_func_map = _module
_register_sharding = _module
_tp_transform = _module
_data_parallel_utils = _module
ddp = _module
input_reshard = _module
loss = _module
style = _module
distributions = _module
bernoulli = _module
beta = _module
binomial = _module
categorical = _module
cauchy = _module
chi2 = _module
constraint_registry = _module
constraints = _module
continuous_bernoulli = _module
dirichlet = _module
distribution = _module
exp_family = _module
exponential = _module
fishersnedecor = _module
gamma = _module
geometric = _module
gumbel = _module
half_cauchy = _module
half_normal = _module
independent = _module
inverse_gamma = _module
kl = _module
kumaraswamy = _module
laplace = _module
lkj_cholesky = _module
log_normal = _module
logistic_normal = _module
lowrank_multivariate_normal = _module
mixture_same_family = _module
multinomial = _module
multivariate_normal = _module
negative_binomial = _module
normal = _module
one_hot_categorical = _module
pareto = _module
poisson = _module
relaxed_bernoulli = _module
relaxed_categorical = _module
studentT = _module
transformed_distribution = _module
transforms = _module
uniform = _module
von_mises = _module
weibull = _module
wishart = _module
_remove_auto_functionalized_pass = _module
_remove_effect_tokens_pass = _module
_safeguard = _module
_swap = _module
_trace = _module
_tree_utils = _module
_unlift = _module
custom_obj = _module
decomp_utils = _module
exported_program = _module
graph_signature = _module
unflatten = _module
func = _module
futures = _module
_compatibility = _module
_lazy_graph_module = _module
_pytree = _module
_symbolic_trace = _module
annotate = _module
_backward_state = _module
_config = _module
accelerator_partitioner = _module
const_fold = _module
graph_gradual_typechecker = _module
merge_matmul = _module
meta_tracer = _module
migrate_gradual_types = _module
constraint = _module
constraint_generator = _module
constraint_transformation = _module
operation = _module
transform_to_z3 = _module
z3_types = _module
normalize = _module
optimization = _module
partitioner_utils = _module
proxy_tensor = _module
recording = _module
refinement_types = _module
rewriter = _module
schema_type_annotation = _module
infer_shape = _module
infer_symbol_values = _module
sym_node = _module
symbolic_shapes = _module
unification = _module
dispatch = _module
match = _module
more = _module
multipledispatch = _module
conflict = _module
dispatcher = _module
variadic = _module
unification_tools = _module
unify_refinements = _module
validator = _module
immutable_collections = _module
interpreter = _module
node = _module
operator_schemas = _module
_tensorify_python_scalars = _module
annotate_getitem_nodes = _module
dialect = _module
cse_pass = _module
fake_tensor_prop = _module
graph_drawer = _module
graph_manipulation = _module
graph_transform_observer = _module
infra = _module
partitioner = _module
pass_manager = _module
net_min_base = _module
operator_support = _module
param_fetch = _module
runtime_assert = _module
shape_prop = _module
split_module = _module
split_utils = _module
splitter_base = _module
tests = _module
test_pass_manager = _module
tools_common = _module
fuser_utils = _module
matcher_utils = _module
matcher_with_name_node_map_utils = _module
source_matcher_utils = _module
proxy = _module
subgraph_rewriter = _module
tensor_type = _module
traceback = _module
hub = _module
_async = _module
_await = _module
_builtins = _module
_check = _module
_dataclass_impls = _module
_decomposition_utils = _module
_decompositions = _module
_freeze = _module
_fuser = _module
_ir_utils = _module
_monkeytype_config = _module
_passes = _module
_property_propagation = _module
_pickle = _module
_recursive = _module
_script = _module
_serialization = _module
_shape_functions = _module
_state = _module
annotations = _module
frontend = _module
generate_bytecode = _module
mobile = _module
supported_ops = _module
unsupported_tensor_ops = _module
library = _module
masked = _module
_docs = _module
maskedtensor = _module
_ops_refs = _module
binary = _module
creation = _module
passthrough = _module
reductions = _module
unary = _module
event = _module
mtia = _module
_atfork = _module
pool = _module
queue = _module
spawn = _module
nested_tensor = _module
_reduction = _module
bias = _module
flex_attention = _module
thnn = _module
common_types = _module
grad = _module
adaptive = _module
channelshuffle = _module
container = _module
distance = _module
flatten = _module
fold = _module
instancenorm = _module
module = _module
padding = _module
pixelshuffle = _module
transformer = _module
upsampling = _module
data_parallel = _module
parallel_apply = _module
scatter_gather = _module
parameter = _module
_reference = _module
_deprecation_utils = _module
_expanded_weights = _module
conv_expanded_weights = _module
conv_utils = _module
embedding_expanded_weights = _module
expanded_weights_impl = _module
expanded_weights_utils = _module
group_norm_expanded_weights = _module
instance_norm_expanded_weights = _module
layer_norm_expanded_weights = _module
linear_expanded_weights = _module
_named_member_accessor = _module
_per_sample_grad = _module
clip_grad = _module
convert_parameters = _module
fusion = _module
memory_format = _module
parametrizations = _module
parametrize = _module
prune = _module
spectral_norm = _module
stateless = _module
weight_norm = _module
onnx = _module
_constants = _module
_deprecation = _module
_exporter_states = _module
_flags = _module
_globals = _module
_exporter_legacy = _module
_lazy_import = _module
diagnostics = _module
_diagnostic = _module
_rules = _module
_infra = _module
formatter = _module
sarif = _module
_address = _module
_artifact = _module
_artifact_change = _module
_artifact_content = _module
_artifact_location = _module
_attachment = _module
_code_flow = _module
_configuration_override = _module
_conversion = _module
_edge = _module
_edge_traversal = _module
_exception = _module
_external_properties = _module
_external_property_file_reference = _module
_external_property_file_references = _module
_fix = _module
_graph = _module
_graph_traversal = _module
_invocation = _module
_location = _module
_location_relationship = _module
_logical_location = _module
_message = _module
_multiformat_message_string = _module
_node = _module
_notification = _module
_physical_location = _module
_property_bag = _module
_rectangle = _module
_region = _module
_replacement = _module
_reporting_configuration = _module
_reporting_descriptor = _module
_reporting_descriptor_reference = _module
_reporting_descriptor_relationship = _module
_result = _module
_result_provenance = _module
_run = _module
_run_automation_details = _module
_sarif_log = _module
_special_locations = _module
_stack = _module
_stack_frame = _module
_suppression = _module
_thread_flow = _module
_thread_flow_location = _module
_tool = _module
_tool_component = _module
_tool_component_reference = _module
_translation_metadata = _module
_version_control_details = _module
_web_request = _module
_web_response = _module
exporter = _module
_analysis = _module
_building = _module
_capture_strategies = _module
_compat = _module
_core = _module
_dispatching = _module
_errors = _module
_fx_passes = _module
_ir_passes = _module
_isolated = _module
_onnx_program = _module
_registration = _module
_reporting = _module
_schemas = _module
_tensors = _module
_verification = _module
_pass = _module
analysis = _module
unsupported_nodes = _module
decomposition_skip = _module
decomposition_table = _module
dynamo_graph_extractor = _module
fx_onnx_interpreter = _module
fx_symbolic_graph_extractor = _module
onnxfunction_dispatcher = _module
decomp = _module
functionalization = _module
modularization = _module
readability = _module
type_promotion = _module
virtualization = _module
patcher = _module
registration = _module
serialization = _module
type_utils = _module
io_adapter = _module
jit_utils = _module
onnx_proto_utils = _module
onnxruntime = _module
_onnx_supported_ops = _module
_type_utils = _module
operators = _module
symbolic_caffe2 = _module
symbolic_helper = _module
symbolic_opset10 = _module
symbolic_opset11 = _module
symbolic_opset12 = _module
symbolic_opset13 = _module
symbolic_opset14 = _module
symbolic_opset15 = _module
symbolic_opset16 = _module
symbolic_opset17 = _module
symbolic_opset18 = _module
symbolic_opset19 = _module
symbolic_opset20 = _module
symbolic_opset7 = _module
symbolic_opset8 = _module
symbolic_opset9 = _module
verification = _module
_adafactor = _module
_functional = _module
_multi_tensor = _module
adadelta = _module
adagrad = _module
adam = _module
adamax = _module
adamw = _module
asgd = _module
lbfgs = _module
lr_scheduler = _module
nadam = _module
radam = _module
rmsprop = _module
rprop = _module
sgd = _module
sparse_adam = _module
swa_utils = _module
overrides = _module
_digraph = _module
_directory_reader = _module
_importlib = _module
_mangling = _module
_mock = _module
_package_pickler = _module
_package_unpickler = _module
_stdlib = _module
analyze = _module
find_first_use_of_broken_modules = _module
is_from_package = _module
trace_dependencies = _module
file_structure_representation = _module
find_file_dependencies = _module
glob_group = _module
importer = _module
package_exporter = _module
package_importer = _module
_memory_profiler = _module
_pattern_matcher = _module
itt = _module
profiler = _module
python_tracer = _module
_quantized_conversions = _module
fusion_patterns = _module
quantization_patterns = _module
quantization_types = _module
quasirandom = _module
return_types = _module
signal = _module
windows = _module
_semi_structured_conversions = _module
_semi_structured_ops = _module
_triton_ops = _module
_triton_ops_meta = _module
semi_structured = _module
_comparison = _module
_creation = _module
autocast_test_lists = _module
autograd_function_db = _module
check_kernel_launches = _module
common_cuda = _module
common_device_type = _module
common_dist_composable = _module
common_distributed = _module
common_dtype = _module
common_fsdp = _module
common_jit = _module
common_methods_invocations = _module
common_mkldnn = _module
common_modules = _module
common_nn = _module
common_optimizers = _module
common_pruning = _module
common_quantization = _module
common_quantized = _module
common_subclass = _module
composite_compliance = _module
custom_op_db = _module
custom_tensor = _module
network1 = _module
network2 = _module
dist_utils = _module
_test_ops_common = _module
_test_st_common = _module
test_common = _module
common_dtensor = _module
checkpoint_utils = _module
common_state_dict = _module
ddp_under_dist_autograd_test = _module
distributed_utils = _module
fake_pg = _module
multi_threaded_pg = _module
remote_module_test = _module
dist_autograd_test = _module
dist_optimizer_test = _module
parameter_server_test = _module
reinforcement_learning_rpc_test = _module
faulty_agent_rpc_test = _module
faulty_rpc_agent_test_fixture = _module
rpc_test = _module
rpc_test_faulty = _module
rpc_agent_test_fixture = _module
tensorpipe_rpc_agent_test_fixture = _module
rpc_utils = _module
dynamo_test_failures = _module
fake_config_module = _module
generated = _module
hop_db = _module
hypothesis_utils = _module
inductor_utils = _module
jit_metaprogramming_utils = _module
logging_tensor = _module
opinfo = _module
_masked = _module
refs = _module
optests = _module
autograd_registration = _module
generate_tests = _module
make_fx = _module
quantization_torch_package_models = _module
static_module = _module
future_div = _module
no_future_div = _module
torchbind_impls = _module
triton_utils = _module
two_tensor = _module
torch_version = _module
_backport_slots = _module
_config_module = _module
_content_store = _module
_contextlib = _module
_cpp_extension_versioner = _module
_cxx_pytree = _module
_device = _module
_exposed_in = _module
_foreach_utils = _module
_get_clean_triton = _module
_import_utils = _module
_mode_utils = _module
_ordered_set = _module
_python_dispatch = _module
_stats = _module
_sympy = _module
functions = _module
interp = _module
numbers = _module
singleton_int = _module
solve = _module
symbol = _module
value_ranges = _module
_thunk = _module
_traceback = _module
_triton = _module
_typing_utils = _module
_zip = _module
backcompat = _module
backend_registration = _module
blas_compare_setup = _module
fuzzer = _module
op_benchmark = _module
simple_timeit = _module
spectral_ops_fuzz_test = _module
op_fuzzers = _module
sparse_binary = _module
sparse_unary = _module
spectral = _module
_stubs = _module
cpp_jit = _module
sparse_fuzzer = _module
valgrind_wrapper = _module
timer_interface = _module
bottleneck = _module
bundled_inputs = _module
collect_env = _module
cpp_backtrace = _module
cpp_extension = _module
collate = _module
fetch = _module
pin_memory = _module
signal_handling = _module
backward_compatibility = _module
dataloader = _module
datapipes = _module
_decorator = _module
_hook_iterator = _module
_typing = _module
dataframe = _module
dataframe_wrapper = _module
dataframes = _module
structures = _module
datapipe = _module
callable = _module
combinatorics = _module
combining = _module
filelister = _module
fileopener = _module
grouping = _module
routeddecoder = _module
selecting = _module
sharding = _module
streamreader = _module
decoder = _module
snapshot = _module
dataset = _module
graph_settings = _module
sampler = _module
deterministic = _module
dlpack = _module
file_baton = _module
flop_counter = _module
hipify = _module
cuda_to_hip_mappings = _module
hipify_python = _module
mobile_optimizer = _module
model_dump = _module
model_zoo = _module
module_tracker = _module
show_pickle = _module
tensorboard = _module
_convert_np = _module
_embedding = _module
_onnx_graph = _module
_proto_graph = _module
_pytorch_graph = _module
summary = _module
writer = _module
throughput_benchmark = _module
viz = _module
_cycles = _module
weak = _module
torchgen = _module
ah_tree = _module
merge_data = _module
gen_data_mixed_mm = _module
test_mixed_mm = _module
train_decision_mixedmm = _module
gen_data_mm = _module
train_decision_mm = _module
gen_data_pad_mm = _module
train_decision_pad_mm = _module
train_pad_mm = _module
train_regression_pad_mm = _module
train = _module
train_decision = _module
train_regression = _module
aoti = _module
fallback_ops = _module
meta = _module
translate = _module
signatures = _module
types_base = _module
ufunc = _module
unboxing = _module
code_template = _module
gen_jit_decompositions = _module
dest = _module
lazy_ir = _module
lazy_ts_lowering = _module
native_functions = _module
register_dispatch_key = _module
et_cpp = _module
parse = _module
gen_patterns = _module
gen_aoti_c_shim = _module
gen_backend_stubs = _module
gen_executorch = _module
gen_functionalization_type = _module
gen_lazy_tensor = _module
gen_schema_utils = _module
gen_vmap_plumbing = _module
local = _module
native_function_generation = _module
operator_versions = _module
gen_mobile_upgraders = _module
gen_mobile_upgraders_constant = _module
selective_build = _module
operator = _module
selector = _module
gen_jit_shape_functions = _module
static_runtime = _module
gen_static_runtime_ops = _module
generator = _module
yaml_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, time, torch, torchaudio, torchvision, triton, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


from typing import Optional


import abc


import collections


import copy


import functools


import itertools


import logging


import time


from typing import Any


from typing import Callable


from typing import Generator


from typing import List


from typing import Mapping


from typing import NamedTuple


from typing import Sequence


from typing import Tuple


from typing import Type


from typing import TYPE_CHECKING


import numpy as np


import numpy.typing as npt


import pandas as pd


from scipy.stats import gmean


from scipy.stats import ttest_ind


import torch


import torch._dynamo


import torch._dynamo.utils


import torch._export


import torch.distributed


import torch.multiprocessing as mp


from torch._C import _has_cuda as HAS_CUDA


from torch._C import _has_xpu as HAS_XPU


from torch._dynamo.profiler import fx_insert_profiling


from torch._dynamo.profiler import Profiler


from torch._dynamo.testing import dummy_fx_compile


from torch._dynamo.testing import format_speedup


from torch._dynamo.testing import reset_rng_state


from torch._dynamo.testing import same


from torch._logging.scribe import open_source_signpost


import torch._functorch.config


from torch._functorch.aot_autograd import set_model_name


from torch._inductor import config as inductor_config


from torch._inductor import metrics


from torch._subclasses.fake_tensor import FakeTensorMode


from torch.utils import _pytree as pytree


from torch.utils._pytree import tree_map


from torch.utils._pytree import tree_map_only


from functools import partial


import torch._dynamo as dynamo


import torch.utils._pytree as pytree


from torch._dynamo.testing import reduce_to_scalar_loss


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.profiler import profile


from torch.profiler import ProfilerActivity


from torch.profiler import record_function


import triton


import torch._inductor.config


from torch._inductor.runtime.benchmarking import benchmarker


import torch._dynamo.config


import torch._inductor.config as config


import torch._inductor.config as inductor_config


import inspect


import torch._inductor


from torch._dynamo.backends.cudagraphs import cudagraphs_inner


from torch._inductor.compile_fx import compile_fx


from torch._inductor.utils import timed


from collections import defaultdict


import torch._dynamo.testing


import torch.distributed._composable.fsdp._fsdp_param


import torch.nn.functional as F


from torch import nn


from torch._dynamo.utils import counters


from torch._inductor import comms


from torch._inductor.utils import is_fallback_op


from torch._inductor.utils import run_and_get_code


from torch.distributed._composable.fsdp import fully_shard


from torch.distributed._composable.fsdp._fsdp_common import TrainingState


from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup


from torch.distributed._tensor import init_device_mesh


from torch.testing import FileCheck


from torch.testing._internal.common_distributed import at_least_x_gpu


from torch.testing._internal.common_distributed import skip_if_lt_x_gpu


from torch.testing._internal.common_fsdp import FSDPTest


from torch.testing._internal.common_fsdp import MLP


from torch.testing._internal.common_utils import run_tests


from torch.testing._internal.common_utils import skipIfRocm


from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs


from torch.testing._internal.distributed._tensor.common_dtensor import Transformer


import torch.distributed as dist


import torch.nn as nn


from torch.distributed._composable import checkpoint


from torch.distributed._composable import fully_shard


from torch.distributed.fsdp import ShardingStrategy


from torch.distributed.fsdp.wrap import ModuleWrapPolicy


from torch.testing._internal.common_fsdp import FSDPInitMode


from torch.testing._internal.common_fsdp import TransformerWithSharedParams


from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN


from copy import deepcopy


from torch import _inductor as inductor


from torch._C import FileCheck


from torch._dynamo import compiled_autograd


from torch._inductor.test_case import TestCase as InductorTestCase


from torch._inductor.utils import run_and_get_triton_code


from torch.distributed._composable.replicate import replicate


from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_default_hooks


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed.tensor.parallel import ColwiseParallel


from torch.distributed.tensor.parallel import parallelize_module


from torch.distributed.tensor.parallel import RowwiseParallel


from torch.nn.parallel.distributed import DistributedDataParallel as DDP


from torch.testing._internal.common_distributed import MultiProcessTestCase


from torch.testing._internal.common_distributed import skip_if_rocm_multiprocess


from torch.testing._internal.distributed.fake_pg import FakeStore


from torch.utils.checkpoint import checkpoint


from torch.distributed._tensor import DeviceMesh


from torch.distributed._tensor import DTensor


from torch.distributed._tensor import Partial


from torch.distributed._tensor import Replicate


from torch.distributed._tensor import Shard


from torch.distributed._tensor.placement_types import DTensorSpec


from torch.distributed._tensor.placement_types import TensorMeta


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.tensor.parallel import PrepareModuleInput


from torch.distributed.tensor.parallel import PrepareModuleOutput


from torch.testing._internal.common_utils import instantiate_parametrized_tests


from torch.testing._internal.common_utils import parametrize


from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule


from torch.testing._internal.distributed._tensor.common_dtensor import with_comms


from torch._dynamo.backends.common import aot_autograd


from typing import Dict


from torch import distributed as dist


from torch.distributed.fsdp import BackwardPrefetch


from torch.distributed.fsdp import CPUOffload


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp import StateDictType


from torch.distributed.fsdp._common_utils import clean_tensor_name


from torch.distributed.fsdp._flat_param import _FSDP_SKIP_WRITEBACK_CHECK


from torch.distributed.fsdp._flat_param import _FSDP_USE_FULL_PREC_IN_EVAL


from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES


from torch.distributed.fsdp.wrap import always_wrap_policy


from torch.nn import TransformerDecoderLayer


from torch.nn import TransformerEncoderLayer


from torch.testing._internal.common_cuda import TEST_CUDA


from torch.testing._internal.common_utils import TestCase


from torch._inductor.decomposition import decompositions


from torch._inductor.fx_passes.micro_pipeline_tp import _get_unexposed_collectives


from torch._inductor.fx_passes.micro_pipeline_tp import find_all_gather_patterns


from torch._inductor.fx_passes.micro_pipeline_tp import find_reduce_scatter_patterns


from torch._inductor.fx_passes.micro_pipeline_tp import micro_pipeline_tp_pass


from torch._inductor.fx_passes.post_grad import remove_noop_ops


from torch._inductor.fx_passes.post_grad import view_to_reshape


from torch._inductor.utils import fresh_inductor_cache


from torch.distributed._functional_collectives import all_gather_tensor


from torch.distributed._functional_collectives import reduce_scatter_tensor


from torch.distributed._symmetric_memory import _test_mode


from torch.distributed._tensor.placement_types import Shard


from torch.distributed.distributed_c10d import _get_group_size_by_name


from torch.testing._internal.common_utils import MI300_ARCH


from torch.testing._internal.common_utils import runOnRocmArch


import torch.distributed._functional_collectives as funcol


from torch.distributed._functional_collectives import all_gather_into_tensor_coalesced


from torch.distributed._functional_collectives import all_reduce


from torch.distributed._functional_collectives import all_reduce_coalesced


from torch.distributed._functional_collectives import all_to_all_single


from torch.distributed._functional_collectives import AsyncCollectiveTensor


from torch.distributed._functional_collectives import reduce_scatter_tensor_coalesced


from torch.testing._internal.common_distributed import requires_nccl


import torch._dynamo.logging


import torch._dynamo.test_case


import torch.distributed._functional_collectives as _functional_collectives


from torch._dynamo.utils import same


from torch._inductor import ir


from torch._inductor import scheduler


from torch._inductor.comm_analysis import baseLat


from torch._inductor.comm_analysis import hwLat


from torch._inductor.comm_analysis import llMaxBws


from torch._inductor.comm_analysis import NCCL_ALGO


from torch._inductor.comm_analysis import NCCL_HW


from torch._inductor.comm_analysis import NCCL_PROTO


from torch._inductor.comm_analysis import NVIDIA_GPU_TYPE


from torch.testing._internal.common_distributed import _dynamo_dist_per_rank_init


from torch.testing._internal.common_distributed import DynamoDistributedMultiProcTestCase


import random


import torch.optim as optim


from torch._dynamo import config


from torch._dynamo.backends.distributed import DDPOptimizer


from torch._dynamo.comptime import comptime


from torch._dynamo.testing import collect_results


from torch._higher_order_ops.wrap import tag_activation_checkpoint


from torch.distributed._functional_collectives import _maybe_wrap_tensor


from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from torch.nn.attention.flex_attention import flex_attention


from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION


from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_MEM_EFF_ATTENTION


from torch.testing._internal.common_distributed import DynamoDistributedSingleProcTestCase


from torch.testing._internal.common_distributed import import_transformers_or_skip


from torch.testing._internal.common_utils import requires_cuda


from functools import wraps


import torch.distributed._functional_collectives as ft_c


import torch.distributed._tensor as dt


import torch.distributed.distributed_c10d as c10d


from torch.testing._internal.common_distributed import MultiThreadedTestCase


from torch.testing._internal.common_distributed import TEST_SKIPS


from torch._dynamo.testing import CompileCounter


from torch._inductor.compile_fx import compile_fx as inductor_compile_fx


from torch.distributed.distributed_c10d import GroupMember


from torch.fx.experimental.proxy_tensor import make_fx


import math


import torch._prims_common as utils


from torch._dynamo.test_case import TestCase


from torch._inductor import config


from torch.library import _scoped_library


from torch.library import Library


from math import prod


import torch._functorch.config as config


from torch.testing._internal.common_utils import TEST_WITH_ROCM


from torch.utils._triton import has_triton


from torch.utils.flop_counter import FlopCounterMode


from torch.utils.flop_counter import register_flop_formula


from torch._higher_order_ops.associative_scan import associative_scan


from torch._higher_order_ops.while_loop import while_loop


from torch._subclasses.functional_tensor import CppFunctionalizeAPI


from torch._subclasses.functional_tensor import FunctionalTensor


from torch._subclasses.functional_tensor import FunctionalTensorMode


from torch._subclasses.functional_tensor import PythonFunctionalizeAPI


from torch.testing._internal.common_cuda import SM70OrLater


from torch.testing._internal.common_quantization import skipIfNoDynamoSupport


from torch.testing._internal.common_utils import decorateIf


from torch.testing._internal.common_utils import IS_WINDOWS


from torch.testing._internal.common_utils import skipIfCrossRef


from torch.testing._internal.common_utils import skipIfTorchDynamo


from torch.testing._internal.common_utils import TEST_WITH_CROSSREF


from torch.testing._internal.common_utils import TEST_WITH_TORCHDYNAMO


from torch.testing._internal.common_utils import xfailIfTorchDynamo


from collections import deque


import torch._functorch


import torch._inductor.decomposition


from torch._functorch.aot_autograd import aot_export_module


from torch._higher_order_ops.effects import with_effects


from torch._higher_order_ops.torchbind import enable_torchbind_tracing


from torch.testing._internal.common_cuda import _get_torch_cuda_version


from torch.testing._internal.common_cuda import SM80OrLater


from torch.testing._internal.common_utils import TEST_CUDA


from torch.testing._internal.torchbind_impls import init_torchbind_implementations


from torch.testing._internal.two_tensor import TwoTensor


from torch._inductor.codegen import triton


from torch._inductor.codegen import wrapper


from torch._inductor.codegen.common import DeviceOpOverrides


from torch._inductor.scheduler import BaseScheduling


from torch._inductor.remote_cache import RemoteCacheBackend


import types


from torch._dynamo import config as dynamo_config


from torch._dynamo.testing import rand_strided


from torch._inductor.runtime.runtime_utils import cache_dir


from torch._inductor.test_case import TestCase


from torch._inductor.utils import run_and_get_cpp_code


from torch.export import Dim


from torch.export import export


from torch.testing._internal import common_utils


from torch.testing._internal.common_cuda import SM90OrLater


from torch.testing._internal.common_quantization import skip_if_no_torchvision


from torch.testing._internal.common_quantization import skipIfNoFBGEMM


from torch.testing._internal.common_utils import DeterministicGuard


from torch.testing._internal.common_utils import find_library_location


from torch.testing._internal.common_utils import IS_CI


from torch.testing._internal.common_utils import IS_FBCODE


from torch.testing._internal.common_utils import IS_MACOS


from torch.testing._internal.common_utils import IS_SANDCASTLE


from torch.testing._internal.logging_utils import LoggingTestCase


from torch.testing._internal.logging_utils import make_logging_test


from torch._inductor.codegen.triton import TritonScheduling


from torch._inductor.test_operators import realize


from torch._inductor.utils import is_big_gpu


from torch.testing._internal.common_utils import slowTest


from torch.testing._internal.common_utils import TEST_WITH_ASAN


from torch._inductor.scheduler import Scheduler


from typing import Union


from torch._dynamo import reset


from torch._inductor.async_compile import AsyncCompile


from torch._inductor.codecache import cuda_compile_command


from torch._inductor.codecache import CUDACodeCache


from torch._inductor.codecache import FxGraphCachePickler


from torch._inductor.codecache import FxGraphHashDetails


from torch._inductor.codecache import PyCodeCache


from torch._inductor.codecache import TensorMetadata


from torch._inductor.codecache import TensorMetadataAndValues


from torch._inductor.graph import GraphLowering


from torch._inductor.test_case import run_tests


from torch._inductor.utils import clear_inductor_caches


from torch.testing._internal.common_device_type import largeTensorTest


from torch._inductor.codegen import triton_utils


from torch._inductor.codegen.common import SizeArg


from torch._inductor.virtualized import V


import queue


import re


from torch._dynamo.backends.debugging import aot_eager


from torch._dynamo.device_interface import get_interface_for_device


from torch.testing._internal.common_utils import skipIfWindows


from torch.testing._internal.logging_utils import logs_to_string


import torch._inductor.cudagraph_trees


import torch.optim.lr_scheduler


from torch.optim import Adadelta


from torch.optim import Adagrad


from torch.optim import Adam


from torch.optim import Adamax


from torch.optim import AdamW


from torch.optim import ASGD


from torch.optim import NAdam


from torch.optim import RAdam


from torch.optim import RMSprop


from torch.optim import Rprop


from torch.optim import SGD


from torch.optim import SparseAdam


from torch.optim.lr_scheduler import ChainedScheduler


from torch.optim.lr_scheduler import ConstantLR


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from torch.optim.lr_scheduler import CyclicLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import LinearLR


from torch.optim.lr_scheduler import MultiplicativeLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import OneCycleLR


from torch.optim.lr_scheduler import PolynomialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from torch.testing._internal.common_device_type import instantiate_device_type_tests


from torch.testing._internal.common_device_type import skipCUDAIf


from torch.testing._internal.common_device_type import skipXPUIf


from torch._inductor.runtime.hints import TRITON_MAX_BLOCK


from torch.testing._internal.common_utils import IS_LINUX


from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner


import torch._dynamo.config as dynamo_config


import torch.backends.cuda


from torch._dynamo.debug_utils import same_two_models


from torch._inductor.compile_fx import compile_fx_inner


from torch._inductor.runtime.hints import DeviceProperties


from torch._inductor.utils import run_and_get_graph_lowering


from torch._inductor.utils import run_fw_bw_and_get_code


from torch.testing._internal.common_cuda import TEST_MULTIGPU


from torch.testing._internal.common_utils import freeze_rng_state


import warnings


from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache


from torch._inductor.codecache import FxGraphCache


from torch._inductor.cudagraph_trees import cudagraphify_impl as tree_cudagraphify_impl


from torch._inductor.cudagraph_utils import FunctionID


from torch.testing._internal.common_utils import TEST_CUDA_GRAPH


from torch.utils._python_dispatch import TorchDispatchMode


import torch._inductor.async_compile


from torch._inductor.codecache import HalideCodeCache


from torch._inductor.runtime.hints import HalideInputSpec


from torch._inductor.runtime.hints import HalideMeta


from torch._inductor.utils import parallel_num_threads


from torch._inductor.codegen.cpp import cexpr


from torch._inductor.codegen.triton import texpr


from torch._inductor.codegen.wrapper import pexpr


from torch._inductor.sizevars import SizeVarAllocator


from torch.utils._sympy.functions import FloorDiv


from torch.utils._sympy.functions import ModularIndexing


from torch.utils._sympy.functions import RoundDecimal


from torch.utils._sympy.functions import RoundToInt


from torch._inductor.utils import override_lowering


from torch.testing._internal.common_utils import skipIfXpu


from torch import Tensor


from torch._higher_order_ops.auto_functionalize import auto_functionalized


from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2


from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core


from torch.testing._internal.common_utils import subtest


from torch.testing._internal.common_device_type import expectedFailureXPU


from torch._inductor.scheduler import SchedulerNode


from torch._inductor.utils import sympy_index_symbol


from torch._inductor.virtualized import ops


from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8


from torch import multiprocessing as mp


from torch._dynamo.exc import BackendCompilerFailed


from torch._inductor.autotune_process import BenchmarkRequest


from torch._inductor.autotune_process import CUDA_VISIBLE_DEVICES


from torch._inductor.autotune_process import TuningProcessPool


from torch._inductor.ir import Buffer


from torch._inductor.ir import ChoiceCaller


from torch._inductor.ir import FixedLayout


from torch._inductor.kernel.mm_plus_mm import aten_mm_plus_mm


from torch._inductor.select_algorithm import AlgorithmSelectorCache


from torch._inductor.select_algorithm import TritonTemplateCaller


from torch._inductor.utils import collect_defined_kernels


from torch._inductor.wrapper_benchmark import get_kernel_category_by_source_code


from torch._dynamo.test_minifier_common import MinifierTestBase


from torch.testing._internal.common_utils import IS_JETSON


from torch._inductor import test_operators


from torch._inductor.codegen.multi_kernel import MultiKernelCall


from torch.nn import functional as F


from torch.testing import make_tensor


from torch._inductor.fx_passes.pad_mm import get_alignment_size


from torch._inductor.fx_passes.pad_mm import get_pad_cache


from torch._inductor.fx_passes.pad_mm import get_padded_length


from torch._inductor.fx_passes.pad_mm import should_pad_common


from torch._dynamo.convert_frame import maybe_cprofile


from torch._dynamo.test_case import run_tests


from torch._inductor.fx_passes import pad_mm as pad_mm_pass


from torch._inductor.utils import ceildiv


from torch.testing._internal.common_utils import serialTest


import torch._inductor.fx_passes.post_grad


from torch._dynamo.utils import count_calls


from torch._higher_order_ops.out_dtype import out_dtype


from torch._inductor.fx_passes import joint_graph


from torch._inductor.pattern_matcher import Arg


from torch._inductor.pattern_matcher import CallFunction


from torch._inductor.pattern_matcher import gen_pattern


from torch._inductor.pattern_matcher import is_mutation_op


from torch._inductor.pattern_matcher import KeywordArg


from torch._inductor.pattern_matcher import Match


from torch._inductor.pattern_matcher import PatternMatcherPass


from torch._inductor.pattern_matcher import PatternPrettyPrinter


from torch._inductor.pattern_matcher import register_graph_pattern


from torch._inductor.pattern_matcher import stable_topological_sort


import torch.autograd


import torch._inductor.test_case


import torch._inductor.utils


from torch.testing._internal.common_utils import TemporaryFileName


import torch._inductor.select_algorithm as select_algorithm


from torch._dynamo.testing import expectedFailureDynamicWrapper


from torch._inductor.autotune_process import TritonBenchmarkRequest


import typing


import torch._inductor.aoti_eager


from torch._dispatch.python import enable_python_dispatcher


from torch._dynamo.debug_utils import aot_graph_input_parser


from torch._dynamo.testing import CompileCounterWithBackend


from torch._dynamo.testing import expectedFailureCodegenDynamic


from torch._dynamo.testing import skipIfPy312


from torch._dynamo.utils import ifdynstaticdefault


from torch._inductor.aoti_eager import aoti_compile_with_persistent_cache


from torch._inductor.aoti_eager import aoti_eager_cache_dir


from torch._inductor.aoti_eager import load_aoti_eager_cache


from torch._inductor.codegen.common import DataTypePropagation


from torch._inductor.codegen.common import OptimizationContext


from torch._inductor.fx_passes import pad_mm


from torch._inductor.utils import add_scheduler_init_hook


from torch._prims_common import is_integer_dtype


from torch.testing._internal.common_cuda import TEST_CUDNN


from torch.testing._internal.common_cuda import with_tf32_off


from torch.testing._internal.common_device_type import _has_sufficient_memory


from torch.testing._internal.common_dtype import all_types


from torch.testing._internal.common_dtype import get_all_dtypes


from torch.testing._internal.common_utils import IS_X86


from torch.testing._internal.common_utils import skipIfNNModuleInlined


from torch.utils._pytree import tree_flatten


from torch.utils._pytree import tree_unflatten


from torch.utils.weak import WeakTensorKeyDictionary


from torch._inductor import cpu_vec_isa


from torch._inductor.compile_fx import complex_memory_overlap


from torch._inductor.utils import has_torchvision_roi_align


import torch.library


from torch._dynamo.testing import make_test_cls_with_patches


from torch._inductor.codegen.common import device_codegens


from torch._inductor.codegen.common import register_backend_for_device


from torch._inductor.codegen.cpp import CppScheduling


from torch.testing._internal.common_device_type import onlyCPU


from torch.testing._internal.common_device_type import onlyOn


from torch.testing._internal.common_utils import IS_ARM64


from torch.testing._internal.common_utils import TEST_CUDA_MEM_LEAK_CHECK


from enum import Enum


from torch._subclasses.fake_tensor import DataDependentOutputException


from torch._subclasses.fake_tensor import DynamicOutputShapeException


from torch.testing._internal.common_device_type import onlyNativeDeviceTypes


from torch.testing._internal.common_device_type import OpDTypes


from torch.testing._internal.common_device_type import ops


from torch.testing._internal.common_device_type import skipCPUIf


from torch.testing._internal.common_utils import dtype_abbrs


from torch.testing._internal.common_utils import skipCUDAMemoryLeakCheckIf


from torch.testing._internal.common_utils import suppress_warnings


from torch.testing._internal.common_utils import TEST_MKL


import torch._functorch.config as functorch_config


from torch._inductor.runtime.runtime_utils import is_power_of_2


import string


import torch.utils.cpp_extension


from torch._dynamo import device_interface


from torch._inductor.codegen.common import get_scheduling_for_device


from torch._inductor.codegen.common import get_wrapper_codegen_for_device


from torch._inductor.codegen.common import register_device_op_overrides


from torch._inductor.utils import get_triton_code


from torch._inductor.runtime.hints import AutotuneHint


from torch._inductor.runtime.hints import HeuristicType


from torch._inductor.runtime.triton_helpers import math as tl_math


from torch._inductor.runtime.triton_heuristics import autotune_hints_to_configs


from torch._inductor.runtime.triton_heuristics import CachingAutotuner


from torch._inductor.runtime.triton_heuristics import triton_config


from torch._higher_order_ops.triton_kernel_wrap import generate_ttir


from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional


from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation


from torch._library import capture_triton


from torch.utils._triton import has_triton_package


from torch import _dynamo as torchdynamo


from torch.autograd import _record_function_with_args_enter


from torch.autograd import _record_function_with_args_exit


from torch.profiler import ExecutionTraceObserver


from torch.profiler import kineto_available


from torch.profiler import supported_activities


from torch.testing._internal.common_utils import skipIfHpu


from torch.testing._internal.common_utils import TEST_HPU


from itertools import product


from random import randint


import torch.cuda


from torch import inf


from torch import nan


from torch.cuda._memory_viz import _profile_to_snapshot


from torch.cuda._memory_viz import profile_plot


from torch.cuda._memory_viz import segment_plot


from torch.cuda._memory_viz import trace_plot


from torch.testing._internal.autocast_test_lists import AutocastTestLists


from torch.testing._internal.autocast_test_lists import TestAutocast


from torch.testing._internal.common_cuda import _create_scaling_case


from torch.testing._internal.common_device_type import onlyCUDA


from torch.testing._internal.common_utils import EXPANDABLE_SEGMENTS


from torch.testing._internal.common_utils import gcIfJetson


from torch.testing._internal.common_utils import get_cycles_per_ms


from torch.testing._internal.common_utils import load_tests


from torch.testing._internal.common_utils import NO_MULTIPROCESSING_SPAWN


from torch.testing._internal.common_utils import skipCUDANonDefaultStreamIf


from torch.testing._internal.common_utils import TEST_NUMPY


from torch.utils.checkpoint import checkpoint_sequential


from torch.utils.viz._cycles import observe_tensor_cycles


from torch.testing._internal.common_cuda import SM53OrLater


from torch.testing._internal.common_cuda import TEST_CUSPARSE_GENERIC


from torch.testing._internal.common_utils import TEST_WITH_TORCHINDUCTOR


from torch.testing._internal.common_utils import TEST_CUDA_CUDSS


from torch.testing._internal.common_utils import TEST_SCIPY


from torch.testing._internal.common_utils import coalescedonoff


from torch.testing._internal.common_utils import IS_REMOTE_GPU


from torch.testing._internal.common_device_type import dtypes


from torch.testing._internal.common_device_type import dtypesIfCUDA


from torch.testing._internal.common_device_type import skipCUDAIfNoSparseGeneric


from torch.testing._internal.common_device_type import precisionOverride


from torch.testing._internal.common_device_type import skipMeta


from torch.testing._internal.common_device_type import skipCPUIfNoMklSparse


from torch.testing._internal.common_device_type import skipCUDAIfRocmVersionLessThan


from torch.testing._internal.common_dtype import floating_types


from torch.testing._internal.common_dtype import all_types_and_complex_and


from torch.testing._internal.common_dtype import floating_and_complex_types


from torch.testing._internal.common_dtype import floating_types_and


from torch.testing._internal.common_dtype import all_types_and_complex


from torch.testing._internal.common_dtype import floating_and_complex_types_and


from torch.sparse import SparseSemiStructuredTensor


from torch.sparse import SparseSemiStructuredTensorCUSPARSELT


from torch.sparse import SparseSemiStructuredTensorCUTLASS


from torch.sparse import to_sparse_semi_structured


from torch.sparse._semi_structured_conversions import sparse_semi_structured_from_dense_cutlass


from torch.sparse._semi_structured_conversions import _sparse_semi_structured_tile


from torch.sparse._semi_structured_conversions import _compute_compressed_swizzled_bitmask


from typing import Iterator


from torch.testing._internal.common_utils import dtype_name


from torch.testing._internal.common_device_type import PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY


from torch.testing._internal.common_device_type import PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY


from torch.testing._internal.common_device_type import get_device_type_test_bases


from torch.testing._internal.common_device_type import deviceCountAtLeast


from torch.testing._internal.common_device_type import expectedFailureMeta


from torch.testing._internal.opinfo.core import SampleInput


from torch.testing._internal.opinfo.core import DecorateInfo


from torch.testing._internal.opinfo.core import OpInfo


from collections import namedtuple


from torch.nn.functional import scaled_dot_product_attention


from torch.nn.attention import sdpa_kernel


from torch.nn.attention import SDPBackend


from torch.nn.attention.bias import CausalVariant


from torch.nn.attention.bias import causal_lower_right


from torch.nn.attention.bias import causal_upper_left


from torch.nn.parameter import Parameter


from torch.testing._internal.common_utils import TEST_FAIRSEQ


from torch.testing._internal.common_utils import set_default_dtype


from torch.testing._internal.common_utils import gradcheck


from torch.testing._internal.common_utils import make_tensor


from torch.testing._internal.common_utils import NOTEST_CPU


from torch.testing._internal.common_utils import TEST_XPU


from torch.testing._internal.common_cuda import IS_JETSON


from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_ATTENTION


from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_CUDNN_ATTENTION


from torch.testing._internal.common_cuda import tf32_on_and_off


from typing import Any as _Any


from typing import Callable as _Callable


from typing import Dict as _Dict


from typing import Optional as _Optional


from typing import overload as _overload


from typing import Set as _Set


from typing import Tuple as _Tuple


from typing import Type as _Type


from typing import TypeVar as _TypeVar


from typing import Union as _Union


from torch._utils import _functionalize_sync as _sync


from torch._utils import _import_dotted_name


from torch._utils import classproperty


from torch._utils_internal import get_file_path


from torch._utils_internal import prepare_multiprocessing_environment


from torch._utils_internal import USE_GLOBAL_DEPS


from torch._utils_internal import USE_RTLD_GLOBAL_WITH_LIBTORCH


from torch import _C as _C


from math import e


from math import inf


from math import nan


from math import pi


from torch._tensor import Tensor


from torch import storage as storage


from torch.storage import _LegacyStorage


from torch.storage import _StorageBase


from torch.storage import _warn_typed_storage_removal


from torch.storage import TypedStorage


from torch.storage import UntypedStorage


from torch import amp as amp


from torch import random as random


from torch import serialization as serialization


from torch._tensor_str import set_printoptions


from torch.amp import autocast


from torch.amp import GradScaler


from torch.random import get_rng_state


from torch.random import initial_seed


from torch.random import manual_seed


from torch.random import seed


from torch.random import set_rng_state


from torch.serialization import load


from torch.serialization import save


from torch._compile import _disable_dynamo


from torch import _VF as _VF


from torch import functional as functional


from torch.functional import *


from torch.autograd import enable_grad as enable_grad


from torch.autograd import inference_mode as inference_mode


from torch.autograd import no_grad as no_grad


from torch.autograd import set_grad_enabled as set_grad_enabled


from torch import __config__ as __config__


from torch import __future__ as __future__


from torch import _awaits as _awaits


from torch import autograd as autograd


from torch import backends as backends


from torch import cpu as cpu


from torch import cuda as cuda


from torch import distributed as distributed


from torch import distributions as distributions


from torch import fft as fft


from torch import futures as futures


from torch import hub as hub


from torch import jit as jit


from torch import linalg as linalg


from torch import mps as mps


from torch import mtia as mtia


from torch import multiprocessing as multiprocessing


from torch import nested as nested


from torch import nn as nn


from torch import optim as optim


from torch import overrides as overrides


from torch import profiler as profiler


from torch import sparse as sparse


from torch import special as special


from torch import testing as testing


from torch import types as types


from torch import utils as utils


from torch import xpu as xpu


from torch.signal import windows as windows


from torch import ao as ao


import torch.nn.intrinsic


import torch.nn.qat


import torch.nn.quantizable


import torch.nn.quantized


from torch import _size_docs


from torch import _storage_docs


from torch import _tensor_docs


from torch import _torch_docs


from torch import _library as _library


from torch import _ops as _ops


from torch._ops import ops as ops


from torch._classes import classes as classes


from torch import quantization as quantization


from torch import quasirandom as quasirandom


from torch.multiprocessing._atfork import register_after_fork


from torch._lobpcg import lobpcg as lobpcg


from torch import masked as masked


from torch._linalg_utils import _symeig as symeig


from torch._linalg_utils import eig


from torch._linalg_utils import lstsq


from torch._linalg_utils import matrix_rank


from torch._linalg_utils import solve


from torch.utils.dlpack import from_dlpack


from torch.utils.dlpack import to_dlpack


from torch import export as export


from torch import func as func


from torch import library as library


from torch import return_types as return_types


from torch._higher_order_ops import cond as cond


from torch._higher_order_ops import while_loop as while_loop


from torch.func import vmap as vmap


import torch.fx.experimental.sym_node


from torch import fx as fx


from torch import _logging


import numbers


from functools import reduce


from itertools import chain


from typing import cast


from typing import Iterable


import torch._meta_registrations


import torch._prims as prims


from torch import sym_float


from torch import sym_int


from torch._decomp import register_decomposition


from torch._prims_common import IntLike


from torch._prims_common import NumberType


from torch._prims_common import suggest_memory_format


from torch._prims_common import TensorLike


from torch._prims_common import TensorSequenceType


from torch._prims_common.wrappers import _maybe_convert_to_dtype


from torch._prims_common.wrappers import _maybe_resize_out


from torch._prims_common.wrappers import _safe_copy_out


from torch._prims_common.wrappers import out_wrapper


from logging import Logger


from torch._dynamo.external_utils import call_backward


from torch._dynamo.external_utils import call_hook


from torch._dynamo.external_utils import FakeCompiledAutogradEngine


from torch._dynamo.source import GetItemSource


from torch._dynamo.source import LocalSource


from torch._dynamo.utils import lazy_format_graph_code


from torch._dynamo.utils import set_locals_to_steal


from torch._logging import getArtifactLogger


from torch._logging import trace_structured


from torch._prims_common import clone_preserve_strides


from torch._subclasses import FakeTensorMode


from torch.fx import GraphModule


from torch.fx.experimental._backward_state import BackwardState


from torch.fx.experimental.proxy_tensor import decompose


from torch.fx.experimental.proxy_tensor import disable_autocast_cache


from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing


from torch.fx.experimental.proxy_tensor import fetch_object_proxy


from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


from torch.fx.experimental.proxy_tensor import PythonKeyTracer


from torch.fx.experimental.proxy_tensor import track_tensor_tree


from torch.fx.experimental.symbolic_shapes import DimDynamic


from torch.fx.experimental.symbolic_shapes import ShapeEnv


from torch.fx.traceback import preserve_node_meta


from torch.fx.traceback import set_stack_trace


from torch.utils._traceback import CapturedTraceback


from collections import Counter


from typing import TypeVar


import torch._subclasses.meta_utils


from torch._prims_common import is_float_dtype


from torch.multiprocessing.reductions import StorageWeakRef


from torch.utils._content_store import ContentStoreReader


from torch.utils._content_store import ContentStoreWriter


from torch.hub import _Faketqdm


from torch.hub import tqdm


from torch._dynamo.trace_rules import _as_posix_path


from torch.utils._traceback import report_compile_source_on_error


import enum


from typing import Set


import torch._inductor.test_operators


import torch.utils._content_store


from torch.utils import _config_module


import uuid


from functools import lru_cache


from types import MethodWrapperType


from typing import ClassVar


from typing import Counter


from typing import DefaultDict


from typing import Deque


from typing import KeysView


from typing import overload


from typing import ValuesView


import torch.fx.experimental.symbolic_shapes


from torch import fx


from torch._C import _instruction_counter


from torch._C import _len_torch_function_stack


from torch._C import _pop_torch_function_stack


from torch._C import _push_on_torch_function_stack


from torch._guards import Source


from torch._guards import TracingContext


from torch._subclasses.meta_utils import is_sparse_compressed


from torch._utils_internal import log_chromium_event_internal


from torch._utils_internal import log_compilation_event


from torch.fx._utils import _format_graph_code


from torch.fx._utils import lazy_format_graph_code


from torch.nn.modules.lazy import LazyModuleMixin


from torch.utils.hooks import RemovableHandle


from torch._subclasses import UnsupportedFakeTensorException


from typing import FrozenSet


from typing import MutableMapping


from torch import SymInt


from torch._guards import GuardSource


from torch._higher_order_ops.torchbind import call_torchbind


from torch._ops import HigherOrderOperator


from torch._subclasses.fake_tensor import FakeTensor


from torch._subclasses.fake_tensor import is_fake


from torch._subclasses.fake_tensor import maybe_get_fake_mode


from torch._subclasses.meta_utils import is_sparse_any


from torch._subclasses.meta_utils import safe_grad


from torch._utils_internal import justknobs_check


from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size


from torch.fx.experimental.symbolic_shapes import RelaxedUnspecConstraint


from torch.fx.experimental.symbolic_shapes import StatefulSymbolicContext


from torch.fx.experimental.symbolic_shapes import SubclassSymbolicContext


from torch.fx.experimental.symbolic_shapes import SymbolicContext


from torch.fx.immutable_collections import immutable_dict


from torch.fx.immutable_collections import immutable_list


from torch.utils._python_dispatch import is_traceable_wrapper_subclass


from torch.utils._sympy.value_ranges import ValueRanges


from torch.utils.weak import TensorWeakRef


from torch._higher_order_ops.triton_kernel_wrap import TritonHOPifier


from typing import NewType


import torch.utils.dlpack


from torch._decomp.decompositions_for_rng import PhiloxStateTracker


from torch._decomp.decompositions_for_rng import rng_decompositions


from torch._dynamo.utils import dynamo_timed


from torch._dynamo.utils import get_chromium_event_logger


from torch._dynamo.utils import preserve_rng_state


from torch._guards import detect_fake_mode


from torch._inductor.utils import BoxedBool


from torch._subclasses import FakeTensor


import torch._inductor.inductor_prims


import torch.fx as fx


from torch.fx.experimental.proxy_tensor import is_sym_node


from torch.fx.experimental.proxy_tensor import py_sym_types


from torch.fx.experimental.sym_node import magic_methods


from torch.fx.experimental.sym_node import method_to_operator


from torch.fx.experimental.symbolic_shapes import find_symbol_binding_fx_nodes


from torch.fx.experimental.symbolic_shapes import free_symbols


from torch.fx.experimental.symbolic_shapes import hint_int


from torch.fx.experimental.symbolic_shapes import is_symbol_binding_fx_node


from torch.fx.passes import graph_drawer


from torch.utils.checkpoint import CheckpointPolicy


from torch.utils._mode_utils import no_dispatch


from torch._C import DispatchKey


from torch._higher_order_ops.utils import _has_potential_branch_input_mutation


from torch._higher_order_ops.utils import autograd_not_implemented


from torch._higher_order_ops.utils import reenter_make_fx


from torch._higher_order_ops.utils import UnsupportedAliasMutationException


from torch.fx.graph_module import GraphModule


from torch.fx.experimental.symbolic_shapes import guard_scalar


import torch.fx


from time import time


from torch._dynamo.device_interface import get_registered_device_interfaces


from torch._inductor.codecache import CodeCacheFuture


from torch._inductor.codecache import CppCodeCache


from torch._inductor.codecache import CppPythonBindingsCodeCache


from torch._inductor.codecache import LambdaFuture


from torch._inductor.codecache import ROCmCodeCache


from torch._inductor.codecache import TritonCodeCache


from torch._inductor.codecache import TritonFuture


from torch._inductor.compile_worker.subproc_pool import _warm_process_pool


from torch._inductor.compile_worker.subproc_pool import AnyPool


from torch._inductor.compile_worker.subproc_pool import SubprocPool


from torch._inductor.compile_worker.watchdog import _async_compile_initializer


from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path


from torch._inductor.runtime.compile_tasks import _worker_compile_triton


from torch import multiprocessing


from torch._inductor.codecache import DLLWrapper


from torch._inductor.codecache import get_hash


from copy import copy


from time import time_ns


from types import ModuleType


from typing import NoReturn


from torch._inductor import exc


from torch._inductor.codegen.cuda import cuda_env


from torch._inductor.codegen.rocm.compile_command import rocm_compile_command


from torch._inductor.codegen.rocm.compile_command import rocm_compiler


from torch._utils_internal import log_cache_bypass


from torch._inductor.cpp_builder import _set_gpu_runtime_env


from torch._inductor.cpp_builder import _transform_cuda_paths


from torch._inductor.cpp_builder import CppBuilder


from torch._inductor.cpp_builder import CppOptions


from torch._inductor.cpp_builder import get_compiler_version_info


from torch._inductor.cpp_builder import get_cpp_compiler


from torch._inductor.cpp_builder import get_name_and_dir_from_output_file_path


from torch._inductor.cpp_builder import normalize_path_separator


from torch._inductor.cpu_vec_isa import pick_vec_isa


from torch._inductor.cudagraph_utils import BoxedDeviceIndex


from torch._inductor.cudagraph_utils import CudagraphCachedInfo


from torch._inductor.cudagraph_utils import log_cudagraph_skip_and_bump_counter


from torch._inductor.runtime.compile_tasks import _module_to_triton_kernel


from torch._inductor.runtime.compile_tasks import _reload_python_module


from torch._inductor.runtime.compile_tasks import _reload_python_module_in_subproc


from torch._inductor.runtime.runtime_utils import default_cache_dir


from torch._inductor.utils import ALIGN_BYTES


from torch._inductor.utils import align_inputs_from_check_idxs


from torch._inductor.utils import clear_on_fresh_inductor_cache


from torch._inductor.utils import is_linux


from torch._inductor.utils import is_windows


from torch._inductor.utils import set_tracing_context_output_strides


from torch._subclasses.fake_tensor import extract_tensor_metadata


from torch._subclasses.fake_tensor import TensorMetadata


from torch.fx.experimental.symbolic_shapes import has_hint


from enum import auto


from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND


from torch.utils._ordered_set import OrderedSet


from torch.utils._sympy.numbers import int_oo


from torch.utils._sympy.symbol import free_symbol_is_type


from torch.utils._sympy.symbol import symbol_is_type


from torch.utils._sympy.symbol import SymT


from torch.utils._sympy.value_ranges import bound_sympy


from torch.utils._sympy.value_ranges import ValueRangeAnalysis


from torch._inductor import dependencies


from torch.utils._sympy.functions import CeilDiv


from itertools import count


import torch._ops


from torch.fx.experimental.symbolic_shapes import ConvertIntKey


from torch.fx.experimental.symbolic_shapes import DivideByKey


from torch.fx.experimental.symbolic_shapes import SymTypes


from itertools import zip_longest


from torch import dtype as torch_dtype


from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name


from torch._inductor.runtime.triton_heuristics import grid as default_grid_fn


from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu


import torch._logging


from torch._inductor.metrics import get_metric_table


from torch._inductor.metrics import is_metric_table_enabled


from typing import no_type_check


from torch.utils._sympy.functions import Identity


from torch.utils._sympy.symbol import prefix_str


import torch._inductor.metrics as metrics


import torch._inductor.runtime.hints


from torch._inductor.codegen.simd import IterationRangesRoot


from torch._inductor.codegen.triton import triton_compute_type


from torch._inductor.codegen.triton import TritonKernel


from torch._inductor.runtime.triton_heuristics import split_scan_grid


from torch._prims_common import prod


from torch._inductor.codegen.debug_utils import DebugPrinterManager


from torch._inductor.codegen.multi_kernel import MultiKernelState


from torch.fx.node import _get_qualified_name


from torch.utils._sympy.singleton_int import SingletonInt


from typing import ContextManager


from torch._dynamo import logging as dynamo_logging


from torch._dynamo import utils as dynamo_utils


from torch._dynamo.repro.after_aot import wrap_compiler_debug


from torch._dynamo.utils import detect_fake_mode


from torch._dynamo.utils import flatten_graph_inputs


from torch._functorch import config as functorch_config


from torch._functorch.aot_autograd import make_boxed_func


from torch._inductor.codecache import _StrideExprStr


from torch._inductor.codecache import code_hash


from torch._inductor.codecache import CompiledFxGraph


from torch._inductor.cudagraph_utils import get_placeholder_info


from torch._inductor.cudagraph_utils import PlaceholderInfo


from torch._inductor.debug import save_args_for_compile_fx_inner


from torch._inductor.utils import count_tangents


from torch._inductor.utils import InputType


from torch._inductor.utils import is_gpu


from torch._inductor.utils import should_assume_input_aligned


from torch._inductor.utils import tensor_is_aligned


from torch._utils_internal import compile_time_strobelight_meta


from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols


from torch.fx.experimental.symbolic_shapes import SymExprPrinter


from torch.fx.passes.fake_tensor_prop import FakeTensorProp


from torch.monitor import _WaitCounter


from torch._inductor.async_compile import pre_fork_setup


from torch._inductor.compile_worker.subproc_pool import SubprocMain


from torch.utils._config_module import install_config_module


from torch._inductor.cpu_vec_isa import invalid_vec_isa


from torch._inductor.cpu_vec_isa import VecISA


from torch.torch_version import TorchVersion


from torch._dynamo.mutation_guard import GenerationTracker


from torch._inductor.compile_fx import align_inputs_from_check_idxs


from torch._inductor.compile_fx import copy_misaligned_inputs


from torch._inductor.compile_fx import get_expanded_dims


from torch._inductor.compile_fx import get_input_idxs_to_check


from torch._inductor.compile_fx import index_expanded_dims


from torch._inductor.compile_fx import remove_unaligned_input_idxs


from torch._inductor.compile_fx import static_input


from torch._inductor.cudagraph_utils import check_for_mutation


from torch._inductor.cudagraph_utils import CheckInvariantStatus


from torch._inductor.cudagraph_utils import log_data_ptr_mismatch


from torch._inductor.cudagraph_utils import maybe_warning_due_to_dynamic_shape


from torch._inductor.cudagraph_utils import ModelType


from torch._inductor.cudagraph_utils import OutputType


from torch._inductor.cudagraph_utils import WrappedFunction


import torch._inductor as inductor


from torch._dynamo.utils import optimus_scuba_log


from torch._prims_common import is_boolean_dtype


from torch._prims_common import is_expandable_to


from torch._utils_internal import upload_graph


from torch.fx.experimental.symbolic_shapes import statically_known_true


from torch.fx.experimental.symbolic_shapes import sym_eq


from torch.fx.passes.graph_transform_observer import GraphTransformObserver


from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table


from torch._inductor import inductor_prims


from torch._inductor.fx_utils import get_node_storage


from torch._inductor.fx_utils import is_node_realized


from torch._inductor.lowering import inplaceable_foreach_ops as inplaceable_foreach_ops_lowerings


from torch.fx.passes.reinplace import _is_view_op


from torch import device


from torch._decomp import get_decompositions


from torch._dynamo.utils import defake


from torch._logging import LazyString


from torch._prims_common import make_channels_last_strides_for


from torch.fx.experimental.symbolic_shapes import has_free_symbols


from torch.fx.experimental.symbolic_shapes import resolve_unbacked_bindings


from torch.fx.experimental.symbolic_shapes import RuntimeAssert


from torch.fx.experimental.symbolic_shapes import SympyBoolean


from torch.fx.graph import Graph


from torch.fx.node import Node


from torch._inductor.codecache import output_code_log


from typing import Literal


import torch._export.serde.schema as export_schema


from torch._dynamo.utils import identity


from torch._export.serde.serialize import GraphModuleSerializer


from torch._higher_order_ops.auto_functionalize import can_auto_functionalize


from torch._prims_common import compute_required_storage_length


from torch._prims_common import StrideType


from torch._subclasses.fake_tensor import get_schema_info


from torch.fx.experimental.symbolic_shapes import CallMethodKey


from torch.fx.experimental.symbolic_shapes import compute_unbacked_bindings


from torch.fx.experimental.symbolic_shapes import rebind_unbacked


from torch.utils._sympy.functions import CleanDiv


from typing import TypedDict


from torch._inductor.kernel.flex_decoding import create_flex_decoding_kernel


from torch._inductor.autoheuristic.autoheuristic import AutoHeuristicSelectAlgorithm


from torch._inductor.autoheuristic.autoheuristic_utils import AHContext


from torch._inductor.autoheuristic.autoheuristic_utils import context_add_strides


from torch._inductor.autoheuristic.autoheuristic_utils import context_add_using_tf32


from torch._inductor.autoheuristic.autoheuristic_utils import get_mixedmm_precondition


from torch._inductor.autoheuristic.autoheuristic_utils import mixed_mm_operations


from torch._inductor.autoheuristic.autoheuristic_utils import mm_operations


from torch._inductor.codegen.cpp_gemm_template import CppPackedGemmTemplate


from torch._inductor.select_algorithm import realize_inputs


from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate


import torch.ao.quantization.fx._decomposed


from torch._higher_order_ops.associative_scan import associative_scan_op


from torch._prims_common import canonicalize_dim


from torch._prims_common import canonicalize_dims


from torch._prims_common import check


from torch._prims_common import dtype_to_type


from torch._prims_common import elementwise_dtypes


from torch._prims_common import get_computation_dtype


from torch._prims_common import Number


from torch.utils._sympy.functions import IntTrueDiv


from torch._inductor.utils import get_benchmark_name


from functools import cached_property


import triton.language as tl


from typing import Container


from typing import Generic


from typing import Protocol


from torch.autograd import DeviceType


from torch.autograd.profiler_util import EventList


from torch.fx.passes.shape_prop import ShapeProp


from torch.utils._sympy.symbol import make_symbol


import torch._library.autograd


import torch._library.fake_impl


import torch._library.simple_registry


import torch._library.utils


from torch._library.fake_class_registry import register_fake_class


from torch._library.triton import capture_triton


from torch._library.triton import triton_op


from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


import torch._C


from torch import device as _device


from torch._utils import _dummy_type


from torch._utils import _LazySeedTracker


from torch.types import Device


from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop


from torch._higher_order_ops.utils import _set_compilation_env


from torch.fx.experimental.proxy_tensor import _temp_remove_pre_dispatch_torch_function_mode


from torch.nn.attention._utils import _supported_head_dim


from torch.nn.attention._utils import _validate_sdpa_input


from abc import ABC


from abc import abstractmethod


from warnings import warn


import torch.autograd.profiler as prof


from torch._C import _get_privateuse1_backend_name


from torch._C._profiler import _add_execution_trace_observer


from torch._C._profiler import _disable_execution_trace_observer


from torch._C._profiler import _enable_execution_trace_observer


from torch._C._profiler import _ExperimentalConfig


from torch._C._profiler import _remove_execution_trace_observer


from torch.autograd import kineto_available


from torch.autograd import ProfilerActivity


from torch.profiler._memory_profiler import MemoryProfile


from torch.profiler._memory_profiler import MemoryProfileTimeline


from torch._dynamo.utils import warn_once


from torch._inductor.utils import get_gpu_shared_memory


from torch._inductor.utils import GPU_TYPES


from torch._inductor.utils import get_gpu_type


from torch.testing._internal.common_utils import LazyVal


from types import FunctionType


from typing import SupportsFloat

