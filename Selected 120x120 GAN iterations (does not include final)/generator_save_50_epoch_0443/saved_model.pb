╚м
й∙
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
└
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%═╠L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ў╛	
И
conv2d_transpose_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_55/bias
Б
,conv2d_transpose_55/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_55/bias*
_output_shapes
:*
dtype0
Щ
conv2d_transpose_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameconv2d_transpose_55/kernel
Т
.conv2d_transpose_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_55/kernel*'
_output_shapes
:А*
dtype0
Й
conv2d_transpose_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameconv2d_transpose_54/bias
В
,conv2d_transpose_54/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_54/bias*
_output_shapes	
:А*
dtype0
Ъ
conv2d_transpose_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*+
shared_nameconv2d_transpose_54/kernel
У
.conv2d_transpose_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_54/kernel*(
_output_shapes
:АА*
dtype0
Й
conv2d_transpose_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameconv2d_transpose_53/bias
В
,conv2d_transpose_53/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_53/bias*
_output_shapes	
:А*
dtype0
Щ
conv2d_transpose_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*+
shared_nameconv2d_transpose_53/kernel
Т
.conv2d_transpose_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_53/kernel*'
_output_shapes
:А@*
dtype0
И
conv2d_transpose_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_52/bias
Б
,conv2d_transpose_52/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_52/bias*
_output_shapes
:@*
dtype0
Ш
conv2d_transpose_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_nameconv2d_transpose_52/kernel
С
.conv2d_transpose_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_52/kernel*&
_output_shapes
:@@*
dtype0
s
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:└p*
shared_namedense_23/bias
l
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes	
:└p*
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d└p* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	d└p*
dtype0
{
serving_default_input_24Placeholder*'
_output_shapes
:         d*
dtype0*
shape:         d
╦
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_24dense_23/kerneldense_23/biasconv2d_transpose_52/kernelconv2d_transpose_52/biasconv2d_transpose_53/kernelconv2d_transpose_53/biasconv2d_transpose_54/kernelconv2d_transpose_54/biasconv2d_transpose_55/kernelconv2d_transpose_55/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8В *,
f'R%
#__inference_signature_wrapper_40707

NoOpNoOp
╖8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Є7
valueш7Bх7 B▐7
┤
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
╚
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op*
О
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 
╚
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
 7_jit_compiled_convolution_op*
О
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
╚
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
 F_jit_compiled_convolution_op*
О
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
╚
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias
 U_jit_compiled_convolution_op*
J
0
1
&2
'3
54
65
D6
E7
S8
T9*
J
0
1
&2
'3
54
65
D6
E7
S8
T9*
* 
░
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
[trace_0
\trace_1
]trace_2
^trace_3* 
6
_trace_0
`trace_1
atrace_2
btrace_3* 
* 

cserving_default* 

0
1*

0
1*
* 
У
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ptrace_0* 

qtrace_0* 

&0
'1*

&0
'1*
* 
У
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

wtrace_0* 

xtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_52/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_52/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
С
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

~trace_0* 

trace_0* 

50
61*

50
61*
* 
Ш
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

Еtrace_0* 

Жtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_53/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_53/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

Мtrace_0* 

Нtrace_0* 

D0
E1*

D0
E1*
* 
Ш
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_54/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_54/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

Ъtrace_0* 

Ыtrace_0* 

S0
T1*

S0
T1*
* 
Ш
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_55/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_55/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
C
0
1
2
3
4
5
6
7
	8*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
щ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp.conv2d_transpose_52/kernel/Read/ReadVariableOp,conv2d_transpose_52/bias/Read/ReadVariableOp.conv2d_transpose_53/kernel/Read/ReadVariableOp,conv2d_transpose_53/bias/Read/ReadVariableOp.conv2d_transpose_54/kernel/Read/ReadVariableOp,conv2d_transpose_54/bias/Read/ReadVariableOp.conv2d_transpose_55/kernel/Read/ReadVariableOp,conv2d_transpose_55/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *'
f"R 
__inference__traced_save_41247
Ь
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_23/kerneldense_23/biasconv2d_transpose_52/kernelconv2d_transpose_52/biasconv2d_transpose_53/kernelconv2d_transpose_53/biasconv2d_transpose_54/kernelconv2d_transpose_54/biasconv2d_transpose_55/kernelconv2d_transpose_55/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В **
f%R#
!__inference__traced_restore_41287гр
т 
Ю
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_41141

inputsD
(conv2d_transpose_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype0▌
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ъ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,                           АБ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
т 
Ю
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_40316

inputsD
(conv2d_transpose_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype0▌
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ъ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,                           АБ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
їВ
Э

H__inference_sequential_23_layer_call_and_return_conditional_losses_40857

inputs:
'dense_23_matmul_readvariableop_resource:	d└p7
(dense_23_biasadd_readvariableop_resource:	└pV
<conv2d_transpose_52_conv2d_transpose_readvariableop_resource:@@A
3conv2d_transpose_52_biasadd_readvariableop_resource:@W
<conv2d_transpose_53_conv2d_transpose_readvariableop_resource:А@B
3conv2d_transpose_53_biasadd_readvariableop_resource:	АX
<conv2d_transpose_54_conv2d_transpose_readvariableop_resource:ААB
3conv2d_transpose_54_biasadd_readvariableop_resource:	АW
<conv2d_transpose_55_conv2d_transpose_readvariableop_resource:АA
3conv2d_transpose_55_biasadd_readvariableop_resource:
identityИв*conv2d_transpose_52/BiasAdd/ReadVariableOpв3conv2d_transpose_52/conv2d_transpose/ReadVariableOpв*conv2d_transpose_53/BiasAdd/ReadVariableOpв3conv2d_transpose_53/conv2d_transpose/ReadVariableOpв*conv2d_transpose_54/BiasAdd/ReadVariableOpв3conv2d_transpose_54/conv2d_transpose/ReadVariableOpв*conv2d_transpose_55/BiasAdd/ReadVariableOpв3conv2d_transpose_55/conv2d_transpose/ReadVariableOpвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpЗ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	d└p*
dtype0|
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └pЕ
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:└p*
dtype0Т
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └pY
reshape_13/ShapeShapedense_23/BiasAdd:output:0*
T0*
_output_shapes
:h
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_13/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@р
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0#reshape_13/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Х
reshape_13/ReshapeReshapedense_23/BiasAdd:output:0!reshape_13/Reshape/shape:output:0*
T0*/
_output_shapes
:         @d
conv2d_transpose_52/ShapeShapereshape_13/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_52/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_52/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_52/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!conv2d_transpose_52/strided_sliceStridedSlice"conv2d_transpose_52/Shape:output:00conv2d_transpose_52/strided_slice/stack:output:02conv2d_transpose_52/strided_slice/stack_1:output:02conv2d_transpose_52/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_52/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_52/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_52/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_52/stackPack*conv2d_transpose_52/strided_slice:output:0$conv2d_transpose_52/stack/1:output:0$conv2d_transpose_52/stack/2:output:0$conv2d_transpose_52/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_52/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_52/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_52/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#conv2d_transpose_52/strided_slice_1StridedSlice"conv2d_transpose_52/stack:output:02conv2d_transpose_52/strided_slice_1/stack:output:04conv2d_transpose_52/strided_slice_1/stack_1:output:04conv2d_transpose_52/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╕
3conv2d_transpose_52/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_52_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ы
$conv2d_transpose_52/conv2d_transposeConv2DBackpropInput"conv2d_transpose_52/stack:output:0;conv2d_transpose_52/conv2d_transpose/ReadVariableOp:value:0reshape_13/Reshape:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ъ
*conv2d_transpose_52/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_52_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0├
conv2d_transpose_52/BiasAddBiasAdd-conv2d_transpose_52/conv2d_transpose:output:02conv2d_transpose_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @|
leaky_re_lu_69/LeakyRelu	LeakyRelu$conv2d_transpose_52/BiasAdd:output:0*/
_output_shapes
:         @o
conv2d_transpose_53/ShapeShape&leaky_re_lu_69/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_53/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_53/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_53/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!conv2d_transpose_53/strided_sliceStridedSlice"conv2d_transpose_53/Shape:output:00conv2d_transpose_53/strided_slice/stack:output:02conv2d_transpose_53/strided_slice/stack_1:output:02conv2d_transpose_53/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_53/stack/1Const*
_output_shapes
: *
dtype0*
value	B :<]
conv2d_transpose_53/stack/2Const*
_output_shapes
: *
dtype0*
value	B :<^
conv2d_transpose_53/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аэ
conv2d_transpose_53/stackPack*conv2d_transpose_53/strided_slice:output:0$conv2d_transpose_53/stack/1:output:0$conv2d_transpose_53/stack/2:output:0$conv2d_transpose_53/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_53/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_53/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_53/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#conv2d_transpose_53/strided_slice_1StridedSlice"conv2d_transpose_53/stack:output:02conv2d_transpose_53/strided_slice_1/stack:output:04conv2d_transpose_53/strided_slice_1/stack_1:output:04conv2d_transpose_53/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╣
3conv2d_transpose_53/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_53_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype0з
$conv2d_transpose_53/conv2d_transposeConv2DBackpropInput"conv2d_transpose_53/stack:output:0;conv2d_transpose_53/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_69/LeakyRelu:activations:0*
T0*0
_output_shapes
:         <<А*
paddingSAME*
strides
Ы
*conv2d_transpose_53/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0─
conv2d_transpose_53/BiasAddBiasAdd-conv2d_transpose_53/conv2d_transpose:output:02conv2d_transpose_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <<А}
leaky_re_lu_70/LeakyRelu	LeakyRelu$conv2d_transpose_53/BiasAdd:output:0*0
_output_shapes
:         <<Аo
conv2d_transpose_54/ShapeShape&leaky_re_lu_70/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_54/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_54/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_54/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!conv2d_transpose_54/strided_sliceStridedSlice"conv2d_transpose_54/Shape:output:00conv2d_transpose_54/strided_slice/stack:output:02conv2d_transpose_54/strided_slice/stack_1:output:02conv2d_transpose_54/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_54/stack/1Const*
_output_shapes
: *
dtype0*
value	B :x]
conv2d_transpose_54/stack/2Const*
_output_shapes
: *
dtype0*
value	B :x^
conv2d_transpose_54/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аэ
conv2d_transpose_54/stackPack*conv2d_transpose_54/strided_slice:output:0$conv2d_transpose_54/stack/1:output:0$conv2d_transpose_54/stack/2:output:0$conv2d_transpose_54/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_54/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_54/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#conv2d_transpose_54/strided_slice_1StridedSlice"conv2d_transpose_54/stack:output:02conv2d_transpose_54/strided_slice_1/stack:output:04conv2d_transpose_54/strided_slice_1/stack_1:output:04conv2d_transpose_54/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask║
3conv2d_transpose_54/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_54_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype0з
$conv2d_transpose_54/conv2d_transposeConv2DBackpropInput"conv2d_transpose_54/stack:output:0;conv2d_transpose_54/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_70/LeakyRelu:activations:0*
T0*0
_output_shapes
:         xxА*
paddingSAME*
strides
Ы
*conv2d_transpose_54/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_54_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0─
conv2d_transpose_54/BiasAddBiasAdd-conv2d_transpose_54/conv2d_transpose:output:02conv2d_transpose_54/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         xxА}
leaky_re_lu_71/LeakyRelu	LeakyRelu$conv2d_transpose_54/BiasAdd:output:0*0
_output_shapes
:         xxАo
conv2d_transpose_55/ShapeShape&leaky_re_lu_71/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_55/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_55/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_55/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!conv2d_transpose_55/strided_sliceStridedSlice"conv2d_transpose_55/Shape:output:00conv2d_transpose_55/strided_slice/stack:output:02conv2d_transpose_55/strided_slice/stack_1:output:02conv2d_transpose_55/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_55/stack/1Const*
_output_shapes
: *
dtype0*
value	B :x]
conv2d_transpose_55/stack/2Const*
_output_shapes
: *
dtype0*
value	B :x]
conv2d_transpose_55/stack/3Const*
_output_shapes
: *
dtype0*
value	B :э
conv2d_transpose_55/stackPack*conv2d_transpose_55/strided_slice:output:0$conv2d_transpose_55/stack/1:output:0$conv2d_transpose_55/stack/2:output:0$conv2d_transpose_55/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_55/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_55/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#conv2d_transpose_55/strided_slice_1StridedSlice"conv2d_transpose_55/stack:output:02conv2d_transpose_55/strided_slice_1/stack:output:04conv2d_transpose_55/strided_slice_1/stack_1:output:04conv2d_transpose_55/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╣
3conv2d_transpose_55/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_55_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype0ж
$conv2d_transpose_55/conv2d_transposeConv2DBackpropInput"conv2d_transpose_55/stack:output:0;conv2d_transpose_55/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_71/LeakyRelu:activations:0*
T0*/
_output_shapes
:         xx*
paddingSAME*
strides
Ъ
*conv2d_transpose_55/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
conv2d_transpose_55/BiasAddBiasAdd-conv2d_transpose_55/conv2d_transpose:output:02conv2d_transpose_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         xxЖ
conv2d_transpose_55/SigmoidSigmoid$conv2d_transpose_55/BiasAdd:output:0*
T0*/
_output_shapes
:         xxv
IdentityIdentityconv2d_transpose_55/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:         xxХ
NoOpNoOp+^conv2d_transpose_52/BiasAdd/ReadVariableOp4^conv2d_transpose_52/conv2d_transpose/ReadVariableOp+^conv2d_transpose_53/BiasAdd/ReadVariableOp4^conv2d_transpose_53/conv2d_transpose/ReadVariableOp+^conv2d_transpose_54/BiasAdd/ReadVariableOp4^conv2d_transpose_54/conv2d_transpose/ReadVariableOp+^conv2d_transpose_55/BiasAdd/ReadVariableOp4^conv2d_transpose_55/conv2d_transpose/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 2X
*conv2d_transpose_52/BiasAdd/ReadVariableOp*conv2d_transpose_52/BiasAdd/ReadVariableOp2j
3conv2d_transpose_52/conv2d_transpose/ReadVariableOp3conv2d_transpose_52/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_53/BiasAdd/ReadVariableOp*conv2d_transpose_53/BiasAdd/ReadVariableOp2j
3conv2d_transpose_53/conv2d_transpose/ReadVariableOp3conv2d_transpose_53/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_54/BiasAdd/ReadVariableOp*conv2d_transpose_54/BiasAdd/ReadVariableOp2j
3conv2d_transpose_54/conv2d_transpose/ReadVariableOp3conv2d_transpose_54/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_55/BiasAdd/ReadVariableOp*conv2d_transpose_55/BiasAdd/ReadVariableOp2j
3conv2d_transpose_55/conv2d_transpose/ReadVariableOp3conv2d_transpose_55/conv2d_transpose/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╫

Ъ
-__inference_sequential_23_layer_call_fn_40757

inputs
unknown:	d└p
	unknown_0:	└p#
	unknown_1:@@
	unknown_2:@$
	unknown_3:А@
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А$
	unknown_7:А
	unknown_8:
identityИвStatefulPartitionedCall╥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40566w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ъЬ
П
 __inference__wrapped_model_40191
input_24H
5sequential_23_dense_23_matmul_readvariableop_resource:	d└pE
6sequential_23_dense_23_biasadd_readvariableop_resource:	└pd
Jsequential_23_conv2d_transpose_52_conv2d_transpose_readvariableop_resource:@@O
Asequential_23_conv2d_transpose_52_biasadd_readvariableop_resource:@e
Jsequential_23_conv2d_transpose_53_conv2d_transpose_readvariableop_resource:А@P
Asequential_23_conv2d_transpose_53_biasadd_readvariableop_resource:	Аf
Jsequential_23_conv2d_transpose_54_conv2d_transpose_readvariableop_resource:ААP
Asequential_23_conv2d_transpose_54_biasadd_readvariableop_resource:	Аe
Jsequential_23_conv2d_transpose_55_conv2d_transpose_readvariableop_resource:АO
Asequential_23_conv2d_transpose_55_biasadd_readvariableop_resource:
identityИв8sequential_23/conv2d_transpose_52/BiasAdd/ReadVariableOpвAsequential_23/conv2d_transpose_52/conv2d_transpose/ReadVariableOpв8sequential_23/conv2d_transpose_53/BiasAdd/ReadVariableOpвAsequential_23/conv2d_transpose_53/conv2d_transpose/ReadVariableOpв8sequential_23/conv2d_transpose_54/BiasAdd/ReadVariableOpвAsequential_23/conv2d_transpose_54/conv2d_transpose/ReadVariableOpв8sequential_23/conv2d_transpose_55/BiasAdd/ReadVariableOpвAsequential_23/conv2d_transpose_55/conv2d_transpose/ReadVariableOpв-sequential_23/dense_23/BiasAdd/ReadVariableOpв,sequential_23/dense_23/MatMul/ReadVariableOpг
,sequential_23/dense_23/MatMul/ReadVariableOpReadVariableOp5sequential_23_dense_23_matmul_readvariableop_resource*
_output_shapes
:	d└p*
dtype0Ъ
sequential_23/dense_23/MatMulMatMulinput_244sequential_23/dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └pб
-sequential_23/dense_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_23_dense_23_biasadd_readvariableop_resource*
_output_shapes	
:└p*
dtype0╝
sequential_23/dense_23/BiasAddBiasAdd'sequential_23/dense_23/MatMul:product:05sequential_23/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └pu
sequential_23/reshape_13/ShapeShape'sequential_23/dense_23/BiasAdd:output:0*
T0*
_output_shapes
:v
,sequential_23/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_23/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_23/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
&sequential_23/reshape_13/strided_sliceStridedSlice'sequential_23/reshape_13/Shape:output:05sequential_23/reshape_13/strided_slice/stack:output:07sequential_23/reshape_13/strided_slice/stack_1:output:07sequential_23/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_23/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_23/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_23/reshape_13/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@ж
&sequential_23/reshape_13/Reshape/shapePack/sequential_23/reshape_13/strided_slice:output:01sequential_23/reshape_13/Reshape/shape/1:output:01sequential_23/reshape_13/Reshape/shape/2:output:01sequential_23/reshape_13/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:┐
 sequential_23/reshape_13/ReshapeReshape'sequential_23/dense_23/BiasAdd:output:0/sequential_23/reshape_13/Reshape/shape:output:0*
T0*/
_output_shapes
:         @А
'sequential_23/conv2d_transpose_52/ShapeShape)sequential_23/reshape_13/Reshape:output:0*
T0*
_output_shapes
:
5sequential_23/conv2d_transpose_52/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7sequential_23/conv2d_transpose_52/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7sequential_23/conv2d_transpose_52/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
/sequential_23/conv2d_transpose_52/strided_sliceStridedSlice0sequential_23/conv2d_transpose_52/Shape:output:0>sequential_23/conv2d_transpose_52/strided_slice/stack:output:0@sequential_23/conv2d_transpose_52/strided_slice/stack_1:output:0@sequential_23/conv2d_transpose_52/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_23/conv2d_transpose_52/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_23/conv2d_transpose_52/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_23/conv2d_transpose_52/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@│
'sequential_23/conv2d_transpose_52/stackPack8sequential_23/conv2d_transpose_52/strided_slice:output:02sequential_23/conv2d_transpose_52/stack/1:output:02sequential_23/conv2d_transpose_52/stack/2:output:02sequential_23/conv2d_transpose_52/stack/3:output:0*
N*
T0*
_output_shapes
:Б
7sequential_23/conv2d_transpose_52/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9sequential_23/conv2d_transpose_52/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9sequential_23/conv2d_transpose_52/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1sequential_23/conv2d_transpose_52/strided_slice_1StridedSlice0sequential_23/conv2d_transpose_52/stack:output:0@sequential_23/conv2d_transpose_52/strided_slice_1/stack:output:0Bsequential_23/conv2d_transpose_52/strided_slice_1/stack_1:output:0Bsequential_23/conv2d_transpose_52/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╘
Asequential_23/conv2d_transpose_52/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_23_conv2d_transpose_52_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0╙
2sequential_23/conv2d_transpose_52/conv2d_transposeConv2DBackpropInput0sequential_23/conv2d_transpose_52/stack:output:0Isequential_23/conv2d_transpose_52/conv2d_transpose/ReadVariableOp:value:0)sequential_23/reshape_13/Reshape:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
╢
8sequential_23/conv2d_transpose_52/BiasAdd/ReadVariableOpReadVariableOpAsequential_23_conv2d_transpose_52_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0э
)sequential_23/conv2d_transpose_52/BiasAddBiasAdd;sequential_23/conv2d_transpose_52/conv2d_transpose:output:0@sequential_23/conv2d_transpose_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @Ш
&sequential_23/leaky_re_lu_69/LeakyRelu	LeakyRelu2sequential_23/conv2d_transpose_52/BiasAdd:output:0*/
_output_shapes
:         @Л
'sequential_23/conv2d_transpose_53/ShapeShape4sequential_23/leaky_re_lu_69/LeakyRelu:activations:0*
T0*
_output_shapes
:
5sequential_23/conv2d_transpose_53/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7sequential_23/conv2d_transpose_53/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7sequential_23/conv2d_transpose_53/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
/sequential_23/conv2d_transpose_53/strided_sliceStridedSlice0sequential_23/conv2d_transpose_53/Shape:output:0>sequential_23/conv2d_transpose_53/strided_slice/stack:output:0@sequential_23/conv2d_transpose_53/strided_slice/stack_1:output:0@sequential_23/conv2d_transpose_53/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_23/conv2d_transpose_53/stack/1Const*
_output_shapes
: *
dtype0*
value	B :<k
)sequential_23/conv2d_transpose_53/stack/2Const*
_output_shapes
: *
dtype0*
value	B :<l
)sequential_23/conv2d_transpose_53/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А│
'sequential_23/conv2d_transpose_53/stackPack8sequential_23/conv2d_transpose_53/strided_slice:output:02sequential_23/conv2d_transpose_53/stack/1:output:02sequential_23/conv2d_transpose_53/stack/2:output:02sequential_23/conv2d_transpose_53/stack/3:output:0*
N*
T0*
_output_shapes
:Б
7sequential_23/conv2d_transpose_53/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9sequential_23/conv2d_transpose_53/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9sequential_23/conv2d_transpose_53/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1sequential_23/conv2d_transpose_53/strided_slice_1StridedSlice0sequential_23/conv2d_transpose_53/stack:output:0@sequential_23/conv2d_transpose_53/strided_slice_1/stack:output:0Bsequential_23/conv2d_transpose_53/strided_slice_1/stack_1:output:0Bsequential_23/conv2d_transpose_53/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╒
Asequential_23/conv2d_transpose_53/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_23_conv2d_transpose_53_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype0▀
2sequential_23/conv2d_transpose_53/conv2d_transposeConv2DBackpropInput0sequential_23/conv2d_transpose_53/stack:output:0Isequential_23/conv2d_transpose_53/conv2d_transpose/ReadVariableOp:value:04sequential_23/leaky_re_lu_69/LeakyRelu:activations:0*
T0*0
_output_shapes
:         <<А*
paddingSAME*
strides
╖
8sequential_23/conv2d_transpose_53/BiasAdd/ReadVariableOpReadVariableOpAsequential_23_conv2d_transpose_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ю
)sequential_23/conv2d_transpose_53/BiasAddBiasAdd;sequential_23/conv2d_transpose_53/conv2d_transpose:output:0@sequential_23/conv2d_transpose_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <<АЩ
&sequential_23/leaky_re_lu_70/LeakyRelu	LeakyRelu2sequential_23/conv2d_transpose_53/BiasAdd:output:0*0
_output_shapes
:         <<АЛ
'sequential_23/conv2d_transpose_54/ShapeShape4sequential_23/leaky_re_lu_70/LeakyRelu:activations:0*
T0*
_output_shapes
:
5sequential_23/conv2d_transpose_54/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7sequential_23/conv2d_transpose_54/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7sequential_23/conv2d_transpose_54/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
/sequential_23/conv2d_transpose_54/strided_sliceStridedSlice0sequential_23/conv2d_transpose_54/Shape:output:0>sequential_23/conv2d_transpose_54/strided_slice/stack:output:0@sequential_23/conv2d_transpose_54/strided_slice/stack_1:output:0@sequential_23/conv2d_transpose_54/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_23/conv2d_transpose_54/stack/1Const*
_output_shapes
: *
dtype0*
value	B :xk
)sequential_23/conv2d_transpose_54/stack/2Const*
_output_shapes
: *
dtype0*
value	B :xl
)sequential_23/conv2d_transpose_54/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А│
'sequential_23/conv2d_transpose_54/stackPack8sequential_23/conv2d_transpose_54/strided_slice:output:02sequential_23/conv2d_transpose_54/stack/1:output:02sequential_23/conv2d_transpose_54/stack/2:output:02sequential_23/conv2d_transpose_54/stack/3:output:0*
N*
T0*
_output_shapes
:Б
7sequential_23/conv2d_transpose_54/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9sequential_23/conv2d_transpose_54/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9sequential_23/conv2d_transpose_54/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1sequential_23/conv2d_transpose_54/strided_slice_1StridedSlice0sequential_23/conv2d_transpose_54/stack:output:0@sequential_23/conv2d_transpose_54/strided_slice_1/stack:output:0Bsequential_23/conv2d_transpose_54/strided_slice_1/stack_1:output:0Bsequential_23/conv2d_transpose_54/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╓
Asequential_23/conv2d_transpose_54/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_23_conv2d_transpose_54_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
2sequential_23/conv2d_transpose_54/conv2d_transposeConv2DBackpropInput0sequential_23/conv2d_transpose_54/stack:output:0Isequential_23/conv2d_transpose_54/conv2d_transpose/ReadVariableOp:value:04sequential_23/leaky_re_lu_70/LeakyRelu:activations:0*
T0*0
_output_shapes
:         xxА*
paddingSAME*
strides
╖
8sequential_23/conv2d_transpose_54/BiasAdd/ReadVariableOpReadVariableOpAsequential_23_conv2d_transpose_54_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ю
)sequential_23/conv2d_transpose_54/BiasAddBiasAdd;sequential_23/conv2d_transpose_54/conv2d_transpose:output:0@sequential_23/conv2d_transpose_54/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         xxАЩ
&sequential_23/leaky_re_lu_71/LeakyRelu	LeakyRelu2sequential_23/conv2d_transpose_54/BiasAdd:output:0*0
_output_shapes
:         xxАЛ
'sequential_23/conv2d_transpose_55/ShapeShape4sequential_23/leaky_re_lu_71/LeakyRelu:activations:0*
T0*
_output_shapes
:
5sequential_23/conv2d_transpose_55/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7sequential_23/conv2d_transpose_55/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7sequential_23/conv2d_transpose_55/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
/sequential_23/conv2d_transpose_55/strided_sliceStridedSlice0sequential_23/conv2d_transpose_55/Shape:output:0>sequential_23/conv2d_transpose_55/strided_slice/stack:output:0@sequential_23/conv2d_transpose_55/strided_slice/stack_1:output:0@sequential_23/conv2d_transpose_55/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_23/conv2d_transpose_55/stack/1Const*
_output_shapes
: *
dtype0*
value	B :xk
)sequential_23/conv2d_transpose_55/stack/2Const*
_output_shapes
: *
dtype0*
value	B :xk
)sequential_23/conv2d_transpose_55/stack/3Const*
_output_shapes
: *
dtype0*
value	B :│
'sequential_23/conv2d_transpose_55/stackPack8sequential_23/conv2d_transpose_55/strided_slice:output:02sequential_23/conv2d_transpose_55/stack/1:output:02sequential_23/conv2d_transpose_55/stack/2:output:02sequential_23/conv2d_transpose_55/stack/3:output:0*
N*
T0*
_output_shapes
:Б
7sequential_23/conv2d_transpose_55/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9sequential_23/conv2d_transpose_55/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9sequential_23/conv2d_transpose_55/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1sequential_23/conv2d_transpose_55/strided_slice_1StridedSlice0sequential_23/conv2d_transpose_55/stack:output:0@sequential_23/conv2d_transpose_55/strided_slice_1/stack:output:0Bsequential_23/conv2d_transpose_55/strided_slice_1/stack_1:output:0Bsequential_23/conv2d_transpose_55/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╒
Asequential_23/conv2d_transpose_55/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_23_conv2d_transpose_55_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype0▐
2sequential_23/conv2d_transpose_55/conv2d_transposeConv2DBackpropInput0sequential_23/conv2d_transpose_55/stack:output:0Isequential_23/conv2d_transpose_55/conv2d_transpose/ReadVariableOp:value:04sequential_23/leaky_re_lu_71/LeakyRelu:activations:0*
T0*/
_output_shapes
:         xx*
paddingSAME*
strides
╢
8sequential_23/conv2d_transpose_55/BiasAdd/ReadVariableOpReadVariableOpAsequential_23_conv2d_transpose_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0э
)sequential_23/conv2d_transpose_55/BiasAddBiasAdd;sequential_23/conv2d_transpose_55/conv2d_transpose:output:0@sequential_23/conv2d_transpose_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         xxв
)sequential_23/conv2d_transpose_55/SigmoidSigmoid2sequential_23/conv2d_transpose_55/BiasAdd:output:0*
T0*/
_output_shapes
:         xxД
IdentityIdentity-sequential_23/conv2d_transpose_55/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:         xxб
NoOpNoOp9^sequential_23/conv2d_transpose_52/BiasAdd/ReadVariableOpB^sequential_23/conv2d_transpose_52/conv2d_transpose/ReadVariableOp9^sequential_23/conv2d_transpose_53/BiasAdd/ReadVariableOpB^sequential_23/conv2d_transpose_53/conv2d_transpose/ReadVariableOp9^sequential_23/conv2d_transpose_54/BiasAdd/ReadVariableOpB^sequential_23/conv2d_transpose_54/conv2d_transpose/ReadVariableOp9^sequential_23/conv2d_transpose_55/BiasAdd/ReadVariableOpB^sequential_23/conv2d_transpose_55/conv2d_transpose/ReadVariableOp.^sequential_23/dense_23/BiasAdd/ReadVariableOp-^sequential_23/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 2t
8sequential_23/conv2d_transpose_52/BiasAdd/ReadVariableOp8sequential_23/conv2d_transpose_52/BiasAdd/ReadVariableOp2Ж
Asequential_23/conv2d_transpose_52/conv2d_transpose/ReadVariableOpAsequential_23/conv2d_transpose_52/conv2d_transpose/ReadVariableOp2t
8sequential_23/conv2d_transpose_53/BiasAdd/ReadVariableOp8sequential_23/conv2d_transpose_53/BiasAdd/ReadVariableOp2Ж
Asequential_23/conv2d_transpose_53/conv2d_transpose/ReadVariableOpAsequential_23/conv2d_transpose_53/conv2d_transpose/ReadVariableOp2t
8sequential_23/conv2d_transpose_54/BiasAdd/ReadVariableOp8sequential_23/conv2d_transpose_54/BiasAdd/ReadVariableOp2Ж
Asequential_23/conv2d_transpose_54/conv2d_transpose/ReadVariableOpAsequential_23/conv2d_transpose_54/conv2d_transpose/ReadVariableOp2t
8sequential_23/conv2d_transpose_55/BiasAdd/ReadVariableOp8sequential_23/conv2d_transpose_55/BiasAdd/ReadVariableOp2Ж
Asequential_23/conv2d_transpose_55/conv2d_transpose/ReadVariableOpAsequential_23/conv2d_transpose_55/conv2d_transpose/ReadVariableOp2^
-sequential_23/dense_23/BiasAdd/ReadVariableOp-sequential_23/dense_23/BiasAdd/ReadVariableOp2\
,sequential_23/dense_23/MatMul/ReadVariableOp,sequential_23/dense_23/MatMul/ReadVariableOp:Q M
'
_output_shapes
:         d
"
_user_specified_name
input_24
У"
Щ
__inference__traced_save_41247
file_prefix.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop9
5savev2_conv2d_transpose_52_kernel_read_readvariableop7
3savev2_conv2d_transpose_52_bias_read_readvariableop9
5savev2_conv2d_transpose_53_kernel_read_readvariableop7
3savev2_conv2d_transpose_53_bias_read_readvariableop9
5savev2_conv2d_transpose_54_kernel_read_readvariableop7
3savev2_conv2d_transpose_54_bias_read_readvariableop9
5savev2_conv2d_transpose_55_kernel_read_readvariableop7
3savev2_conv2d_transpose_55_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ░
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┘
value╧B╠B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B └
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop5savev2_conv2d_transpose_52_kernel_read_readvariableop3savev2_conv2d_transpose_52_bias_read_readvariableop5savev2_conv2d_transpose_53_kernel_read_readvariableop3savev2_conv2d_transpose_53_bias_read_readvariableop5savev2_conv2d_transpose_54_kernel_read_readvariableop3savev2_conv2d_transpose_54_bias_read_readvariableop5savev2_conv2d_transpose_55_kernel_read_readvariableop3savev2_conv2d_transpose_55_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*П
_input_shapes~
|: :	d└p:└p:@@:@:А@:А:АА:А:А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	d└p:!

_output_shapes	
:└p:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:А@:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:-	)
'
_output_shapes
:А: 


_output_shapes
::

_output_shapes
: 
╠
J
.__inference_leaky_re_lu_69_layer_call_fn_41042

inputs
identity┴
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_40417h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
№,
Ф
!__inference__traced_restore_41287
file_prefix3
 assignvariableop_dense_23_kernel:	d└p/
 assignvariableop_1_dense_23_bias:	└pG
-assignvariableop_2_conv2d_transpose_52_kernel:@@9
+assignvariableop_3_conv2d_transpose_52_bias:@H
-assignvariableop_4_conv2d_transpose_53_kernel:А@:
+assignvariableop_5_conv2d_transpose_53_bias:	АI
-assignvariableop_6_conv2d_transpose_54_kernel:АА:
+assignvariableop_7_conv2d_transpose_54_bias:	АH
-assignvariableop_8_conv2d_transpose_55_kernel:А9
+assignvariableop_9_conv2d_transpose_55_bias:
identity_11ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9│
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┘
value╧B╠B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ╒
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_dense_23_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_23_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_2AssignVariableOp-assignvariableop_2_conv2d_transpose_52_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_3AssignVariableOp+assignvariableop_3_conv2d_transpose_52_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_4AssignVariableOp-assignvariableop_4_conv2d_transpose_53_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_5AssignVariableOp+assignvariableop_5_conv2d_transpose_53_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_54_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_54_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_8AssignVariableOp-assignvariableop_8_conv2d_transpose_55_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_9AssignVariableOp+assignvariableop_9_conv2d_transpose_55_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 л
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: Ш
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╫

Ъ
-__inference_sequential_23_layer_call_fn_40732

inputs
unknown:	d└p
	unknown_0:	└p#
	unknown_1:@@
	unknown_2:@$
	unknown_3:А@
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А$
	unknown_7:А
	unknown_8:
identityИвStatefulPartitionedCall╥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40449w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
є
e
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_40417

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╬
a
E__inference_reshape_13_layer_call_and_return_conditional_losses_40995

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         @`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └p:P L
(
_output_shapes
:         └p
 
_user_specified_nameinputs
ў
e
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_40429

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:         <<Аh
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:         <<А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         <<А:X T
0
_output_shapes
:         <<А
 
_user_specified_nameinputs
╬
a
E__inference_reshape_13_layer_call_and_return_conditional_losses_40405

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         @`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └p:P L
(
_output_shapes
:         └p
 
_user_specified_nameinputs
▌

Ь
-__inference_sequential_23_layer_call_fn_40472
input_24
unknown:	d└p
	unknown_0:	└p#
	unknown_1:@@
	unknown_2:@$
	unknown_3:А@
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А$
	unknown_7:А
	unknown_8:
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40449w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         d
"
_user_specified_name
input_24
╪+
Д
H__inference_sequential_23_layer_call_and_return_conditional_losses_40680
input_24!
dense_23_40650:	d└p
dense_23_40652:	└p3
conv2d_transpose_52_40656:@@'
conv2d_transpose_52_40658:@4
conv2d_transpose_53_40662:А@(
conv2d_transpose_53_40664:	А5
conv2d_transpose_54_40668:АА(
conv2d_transpose_54_40670:	А4
conv2d_transpose_55_40674:А'
conv2d_transpose_55_40676:
identityИв+conv2d_transpose_52/StatefulPartitionedCallв+conv2d_transpose_53/StatefulPartitionedCallв+conv2d_transpose_54/StatefulPartitionedCallв+conv2d_transpose_55/StatefulPartitionedCallв dense_23/StatefulPartitionedCallї
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinput_24dense_23_40650dense_23_40652*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └p*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_40385ы
reshape_13/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_40405├
+conv2d_transpose_52/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0conv2d_transpose_52_40656conv2d_transpose_52_40658*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_40228■
leaky_re_lu_69/PartitionedCallPartitionedCall4conv2d_transpose_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_40417╚
+conv2d_transpose_53/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_69/PartitionedCall:output:0conv2d_transpose_53_40662conv2d_transpose_53_40664*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_40272 
leaky_re_lu_70/PartitionedCallPartitionedCall4conv2d_transpose_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_40429╚
+conv2d_transpose_54/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_70/PartitionedCall:output:0conv2d_transpose_54_40668conv2d_transpose_54_40670*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_40316 
leaky_re_lu_71/PartitionedCallPartitionedCall4conv2d_transpose_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_40441╟
+conv2d_transpose_55/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0conv2d_transpose_55_40674conv2d_transpose_55_40676*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_40361Л
IdentityIdentity4conv2d_transpose_55/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xxб
NoOpNoOp,^conv2d_transpose_52/StatefulPartitionedCall,^conv2d_transpose_53/StatefulPartitionedCall,^conv2d_transpose_54/StatefulPartitionedCall,^conv2d_transpose_55/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 2Z
+conv2d_transpose_52/StatefulPartitionedCall+conv2d_transpose_52/StatefulPartitionedCall2Z
+conv2d_transpose_53/StatefulPartitionedCall+conv2d_transpose_53/StatefulPartitionedCall2Z
+conv2d_transpose_54/StatefulPartitionedCall+conv2d_transpose_54/StatefulPartitionedCall2Z
+conv2d_transpose_55/StatefulPartitionedCall+conv2d_transpose_55/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:Q M
'
_output_shapes
:         d
"
_user_specified_name
input_24
╓ 
Ы
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_41037

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ў
e
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_41151

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:         xxАh
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:         xxА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         xxА:X T
0
_output_shapes
:         xxА
 
_user_specified_nameinputs
є
e
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_41047

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
їВ
Э

H__inference_sequential_23_layer_call_and_return_conditional_losses_40957

inputs:
'dense_23_matmul_readvariableop_resource:	d└p7
(dense_23_biasadd_readvariableop_resource:	└pV
<conv2d_transpose_52_conv2d_transpose_readvariableop_resource:@@A
3conv2d_transpose_52_biasadd_readvariableop_resource:@W
<conv2d_transpose_53_conv2d_transpose_readvariableop_resource:А@B
3conv2d_transpose_53_biasadd_readvariableop_resource:	АX
<conv2d_transpose_54_conv2d_transpose_readvariableop_resource:ААB
3conv2d_transpose_54_biasadd_readvariableop_resource:	АW
<conv2d_transpose_55_conv2d_transpose_readvariableop_resource:АA
3conv2d_transpose_55_biasadd_readvariableop_resource:
identityИв*conv2d_transpose_52/BiasAdd/ReadVariableOpв3conv2d_transpose_52/conv2d_transpose/ReadVariableOpв*conv2d_transpose_53/BiasAdd/ReadVariableOpв3conv2d_transpose_53/conv2d_transpose/ReadVariableOpв*conv2d_transpose_54/BiasAdd/ReadVariableOpв3conv2d_transpose_54/conv2d_transpose/ReadVariableOpв*conv2d_transpose_55/BiasAdd/ReadVariableOpв3conv2d_transpose_55/conv2d_transpose/ReadVariableOpвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpЗ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	d└p*
dtype0|
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └pЕ
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:└p*
dtype0Т
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └pY
reshape_13/ShapeShapedense_23/BiasAdd:output:0*
T0*
_output_shapes
:h
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_13/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@р
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0#reshape_13/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Х
reshape_13/ReshapeReshapedense_23/BiasAdd:output:0!reshape_13/Reshape/shape:output:0*
T0*/
_output_shapes
:         @d
conv2d_transpose_52/ShapeShapereshape_13/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_52/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_52/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_52/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!conv2d_transpose_52/strided_sliceStridedSlice"conv2d_transpose_52/Shape:output:00conv2d_transpose_52/strided_slice/stack:output:02conv2d_transpose_52/strided_slice/stack_1:output:02conv2d_transpose_52/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_52/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_52/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_52/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_52/stackPack*conv2d_transpose_52/strided_slice:output:0$conv2d_transpose_52/stack/1:output:0$conv2d_transpose_52/stack/2:output:0$conv2d_transpose_52/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_52/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_52/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_52/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#conv2d_transpose_52/strided_slice_1StridedSlice"conv2d_transpose_52/stack:output:02conv2d_transpose_52/strided_slice_1/stack:output:04conv2d_transpose_52/strided_slice_1/stack_1:output:04conv2d_transpose_52/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╕
3conv2d_transpose_52/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_52_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ы
$conv2d_transpose_52/conv2d_transposeConv2DBackpropInput"conv2d_transpose_52/stack:output:0;conv2d_transpose_52/conv2d_transpose/ReadVariableOp:value:0reshape_13/Reshape:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ъ
*conv2d_transpose_52/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_52_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0├
conv2d_transpose_52/BiasAddBiasAdd-conv2d_transpose_52/conv2d_transpose:output:02conv2d_transpose_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @|
leaky_re_lu_69/LeakyRelu	LeakyRelu$conv2d_transpose_52/BiasAdd:output:0*/
_output_shapes
:         @o
conv2d_transpose_53/ShapeShape&leaky_re_lu_69/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_53/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_53/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_53/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!conv2d_transpose_53/strided_sliceStridedSlice"conv2d_transpose_53/Shape:output:00conv2d_transpose_53/strided_slice/stack:output:02conv2d_transpose_53/strided_slice/stack_1:output:02conv2d_transpose_53/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_53/stack/1Const*
_output_shapes
: *
dtype0*
value	B :<]
conv2d_transpose_53/stack/2Const*
_output_shapes
: *
dtype0*
value	B :<^
conv2d_transpose_53/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аэ
conv2d_transpose_53/stackPack*conv2d_transpose_53/strided_slice:output:0$conv2d_transpose_53/stack/1:output:0$conv2d_transpose_53/stack/2:output:0$conv2d_transpose_53/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_53/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_53/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_53/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#conv2d_transpose_53/strided_slice_1StridedSlice"conv2d_transpose_53/stack:output:02conv2d_transpose_53/strided_slice_1/stack:output:04conv2d_transpose_53/strided_slice_1/stack_1:output:04conv2d_transpose_53/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╣
3conv2d_transpose_53/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_53_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype0з
$conv2d_transpose_53/conv2d_transposeConv2DBackpropInput"conv2d_transpose_53/stack:output:0;conv2d_transpose_53/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_69/LeakyRelu:activations:0*
T0*0
_output_shapes
:         <<А*
paddingSAME*
strides
Ы
*conv2d_transpose_53/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0─
conv2d_transpose_53/BiasAddBiasAdd-conv2d_transpose_53/conv2d_transpose:output:02conv2d_transpose_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <<А}
leaky_re_lu_70/LeakyRelu	LeakyRelu$conv2d_transpose_53/BiasAdd:output:0*0
_output_shapes
:         <<Аo
conv2d_transpose_54/ShapeShape&leaky_re_lu_70/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_54/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_54/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_54/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!conv2d_transpose_54/strided_sliceStridedSlice"conv2d_transpose_54/Shape:output:00conv2d_transpose_54/strided_slice/stack:output:02conv2d_transpose_54/strided_slice/stack_1:output:02conv2d_transpose_54/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_54/stack/1Const*
_output_shapes
: *
dtype0*
value	B :x]
conv2d_transpose_54/stack/2Const*
_output_shapes
: *
dtype0*
value	B :x^
conv2d_transpose_54/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аэ
conv2d_transpose_54/stackPack*conv2d_transpose_54/strided_slice:output:0$conv2d_transpose_54/stack/1:output:0$conv2d_transpose_54/stack/2:output:0$conv2d_transpose_54/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_54/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_54/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#conv2d_transpose_54/strided_slice_1StridedSlice"conv2d_transpose_54/stack:output:02conv2d_transpose_54/strided_slice_1/stack:output:04conv2d_transpose_54/strided_slice_1/stack_1:output:04conv2d_transpose_54/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask║
3conv2d_transpose_54/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_54_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype0з
$conv2d_transpose_54/conv2d_transposeConv2DBackpropInput"conv2d_transpose_54/stack:output:0;conv2d_transpose_54/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_70/LeakyRelu:activations:0*
T0*0
_output_shapes
:         xxА*
paddingSAME*
strides
Ы
*conv2d_transpose_54/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_54_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0─
conv2d_transpose_54/BiasAddBiasAdd-conv2d_transpose_54/conv2d_transpose:output:02conv2d_transpose_54/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         xxА}
leaky_re_lu_71/LeakyRelu	LeakyRelu$conv2d_transpose_54/BiasAdd:output:0*0
_output_shapes
:         xxАo
conv2d_transpose_55/ShapeShape&leaky_re_lu_71/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_55/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_55/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_55/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!conv2d_transpose_55/strided_sliceStridedSlice"conv2d_transpose_55/Shape:output:00conv2d_transpose_55/strided_slice/stack:output:02conv2d_transpose_55/strided_slice/stack_1:output:02conv2d_transpose_55/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_55/stack/1Const*
_output_shapes
: *
dtype0*
value	B :x]
conv2d_transpose_55/stack/2Const*
_output_shapes
: *
dtype0*
value	B :x]
conv2d_transpose_55/stack/3Const*
_output_shapes
: *
dtype0*
value	B :э
conv2d_transpose_55/stackPack*conv2d_transpose_55/strided_slice:output:0$conv2d_transpose_55/stack/1:output:0$conv2d_transpose_55/stack/2:output:0$conv2d_transpose_55/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_55/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_55/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#conv2d_transpose_55/strided_slice_1StridedSlice"conv2d_transpose_55/stack:output:02conv2d_transpose_55/strided_slice_1/stack:output:04conv2d_transpose_55/strided_slice_1/stack_1:output:04conv2d_transpose_55/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╣
3conv2d_transpose_55/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_55_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype0ж
$conv2d_transpose_55/conv2d_transposeConv2DBackpropInput"conv2d_transpose_55/stack:output:0;conv2d_transpose_55/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_71/LeakyRelu:activations:0*
T0*/
_output_shapes
:         xx*
paddingSAME*
strides
Ъ
*conv2d_transpose_55/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
conv2d_transpose_55/BiasAddBiasAdd-conv2d_transpose_55/conv2d_transpose:output:02conv2d_transpose_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         xxЖ
conv2d_transpose_55/SigmoidSigmoid$conv2d_transpose_55/BiasAdd:output:0*
T0*/
_output_shapes
:         xxv
IdentityIdentityconv2d_transpose_55/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:         xxХ
NoOpNoOp+^conv2d_transpose_52/BiasAdd/ReadVariableOp4^conv2d_transpose_52/conv2d_transpose/ReadVariableOp+^conv2d_transpose_53/BiasAdd/ReadVariableOp4^conv2d_transpose_53/conv2d_transpose/ReadVariableOp+^conv2d_transpose_54/BiasAdd/ReadVariableOp4^conv2d_transpose_54/conv2d_transpose/ReadVariableOp+^conv2d_transpose_55/BiasAdd/ReadVariableOp4^conv2d_transpose_55/conv2d_transpose/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 2X
*conv2d_transpose_52/BiasAdd/ReadVariableOp*conv2d_transpose_52/BiasAdd/ReadVariableOp2j
3conv2d_transpose_52/conv2d_transpose/ReadVariableOp3conv2d_transpose_52/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_53/BiasAdd/ReadVariableOp*conv2d_transpose_53/BiasAdd/ReadVariableOp2j
3conv2d_transpose_53/conv2d_transpose/ReadVariableOp3conv2d_transpose_53/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_54/BiasAdd/ReadVariableOp*conv2d_transpose_54/BiasAdd/ReadVariableOp2j
3conv2d_transpose_54/conv2d_transpose/ReadVariableOp3conv2d_transpose_54/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_55/BiasAdd/ReadVariableOp*conv2d_transpose_55/BiasAdd/ReadVariableOp2j
3conv2d_transpose_55/conv2d_transpose/ReadVariableOp3conv2d_transpose_55/conv2d_transpose/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╥+
В
H__inference_sequential_23_layer_call_and_return_conditional_losses_40566

inputs!
dense_23_40536:	d└p
dense_23_40538:	└p3
conv2d_transpose_52_40542:@@'
conv2d_transpose_52_40544:@4
conv2d_transpose_53_40548:А@(
conv2d_transpose_53_40550:	А5
conv2d_transpose_54_40554:АА(
conv2d_transpose_54_40556:	А4
conv2d_transpose_55_40560:А'
conv2d_transpose_55_40562:
identityИв+conv2d_transpose_52/StatefulPartitionedCallв+conv2d_transpose_53/StatefulPartitionedCallв+conv2d_transpose_54/StatefulPartitionedCallв+conv2d_transpose_55/StatefulPartitionedCallв dense_23/StatefulPartitionedCallє
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_40536dense_23_40538*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └p*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_40385ы
reshape_13/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_40405├
+conv2d_transpose_52/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0conv2d_transpose_52_40542conv2d_transpose_52_40544*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_40228■
leaky_re_lu_69/PartitionedCallPartitionedCall4conv2d_transpose_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_40417╚
+conv2d_transpose_53/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_69/PartitionedCall:output:0conv2d_transpose_53_40548conv2d_transpose_53_40550*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_40272 
leaky_re_lu_70/PartitionedCallPartitionedCall4conv2d_transpose_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_40429╚
+conv2d_transpose_54/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_70/PartitionedCall:output:0conv2d_transpose_54_40554conv2d_transpose_54_40556*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_40316 
leaky_re_lu_71/PartitionedCallPartitionedCall4conv2d_transpose_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_40441╟
+conv2d_transpose_55/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0conv2d_transpose_55_40560conv2d_transpose_55_40562*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_40361Л
IdentityIdentity4conv2d_transpose_55/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xxб
NoOpNoOp,^conv2d_transpose_52/StatefulPartitionedCall,^conv2d_transpose_53/StatefulPartitionedCall,^conv2d_transpose_54/StatefulPartitionedCall,^conv2d_transpose_55/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 2Z
+conv2d_transpose_52/StatefulPartitionedCall+conv2d_transpose_52/StatefulPartitionedCall2Z
+conv2d_transpose_53/StatefulPartitionedCall+conv2d_transpose_53/StatefulPartitionedCall2Z
+conv2d_transpose_54/StatefulPartitionedCall+conv2d_transpose_54/StatefulPartitionedCall2Z
+conv2d_transpose_55/StatefulPartitionedCall+conv2d_transpose_55/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
▐ 
Э
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_40272

inputsC
(conv2d_transpose_readvariableop_resource:А@.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype0▌
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ъ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,                           АБ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╠
и
3__inference_conv2d_transpose_52_layer_call_fn_41004

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_40228Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
л

Т
#__inference_signature_wrapper_40707
input_24
unknown:	d└p
	unknown_0:	└p#
	unknown_1:@@
	unknown_2:@$
	unknown_3:А@
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А$
	unknown_7:А
	unknown_8:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8В *)
f$R"
 __inference__wrapped_model_40191w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         d
"
_user_specified_name
input_24
╙
л
3__inference_conv2d_transpose_54_layer_call_fn_41108

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_40316К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╨
J
.__inference_leaky_re_lu_71_layer_call_fn_41146

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_40441i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         xxА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         xxА:X T
0
_output_shapes
:         xxА
 
_user_specified_nameinputs
═	
Ў
C__inference_dense_23_layer_call_and_return_conditional_losses_40976

inputs1
matmul_readvariableop_resource:	d└p.
biasadd_readvariableop_resource:	└p
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d└p*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └ps
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:└p*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └p`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         └pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╧
й
3__inference_conv2d_transpose_55_layer_call_fn_41160

inputs"
unknown:А
	unknown_0:
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_40361Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
═	
Ў
C__inference_dense_23_layer_call_and_return_conditional_losses_40385

inputs1
matmul_readvariableop_resource:	d└p.
biasadd_readvariableop_resource:	└p
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d└p*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └ps
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:└p*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └p`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         └pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╨
к
3__inference_conv2d_transpose_53_layer_call_fn_41056

inputs"
unknown:А@
	unknown_0:	А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_40272К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▌

Ь
-__inference_sequential_23_layer_call_fn_40614
input_24
unknown:	d└p
	unknown_0:	└p#
	unknown_1:@@
	unknown_2:@$
	unknown_3:А@
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А$
	unknown_7:А
	unknown_8:
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40566w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         d
"
_user_specified_name
input_24
╔
Ч
(__inference_dense_23_layer_call_fn_40966

inputs
unknown:	d└p
	unknown_0:	└p
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └p*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_40385p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         └p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ў
e
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_40441

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:         xxАh
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:         xxА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         xxА:X T
0
_output_shapes
:         xxА
 
_user_specified_nameinputs
╨
J
.__inference_leaky_re_lu_70_layer_call_fn_41094

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_40429i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         <<А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         <<А:X T
0
_output_shapes
:         <<А
 
_user_specified_nameinputs
╪+
Д
H__inference_sequential_23_layer_call_and_return_conditional_losses_40647
input_24!
dense_23_40617:	d└p
dense_23_40619:	└p3
conv2d_transpose_52_40623:@@'
conv2d_transpose_52_40625:@4
conv2d_transpose_53_40629:А@(
conv2d_transpose_53_40631:	А5
conv2d_transpose_54_40635:АА(
conv2d_transpose_54_40637:	А4
conv2d_transpose_55_40641:А'
conv2d_transpose_55_40643:
identityИв+conv2d_transpose_52/StatefulPartitionedCallв+conv2d_transpose_53/StatefulPartitionedCallв+conv2d_transpose_54/StatefulPartitionedCallв+conv2d_transpose_55/StatefulPartitionedCallв dense_23/StatefulPartitionedCallї
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinput_24dense_23_40617dense_23_40619*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └p*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_40385ы
reshape_13/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_40405├
+conv2d_transpose_52/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0conv2d_transpose_52_40623conv2d_transpose_52_40625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_40228■
leaky_re_lu_69/PartitionedCallPartitionedCall4conv2d_transpose_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_40417╚
+conv2d_transpose_53/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_69/PartitionedCall:output:0conv2d_transpose_53_40629conv2d_transpose_53_40631*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_40272 
leaky_re_lu_70/PartitionedCallPartitionedCall4conv2d_transpose_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_40429╚
+conv2d_transpose_54/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_70/PartitionedCall:output:0conv2d_transpose_54_40635conv2d_transpose_54_40637*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_40316 
leaky_re_lu_71/PartitionedCallPartitionedCall4conv2d_transpose_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_40441╟
+conv2d_transpose_55/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0conv2d_transpose_55_40641conv2d_transpose_55_40643*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_40361Л
IdentityIdentity4conv2d_transpose_55/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xxб
NoOpNoOp,^conv2d_transpose_52/StatefulPartitionedCall,^conv2d_transpose_53/StatefulPartitionedCall,^conv2d_transpose_54/StatefulPartitionedCall,^conv2d_transpose_55/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 2Z
+conv2d_transpose_52/StatefulPartitionedCall+conv2d_transpose_52/StatefulPartitionedCall2Z
+conv2d_transpose_53/StatefulPartitionedCall+conv2d_transpose_53/StatefulPartitionedCall2Z
+conv2d_transpose_54/StatefulPartitionedCall+conv2d_transpose_54/StatefulPartitionedCall2Z
+conv2d_transpose_55/StatefulPartitionedCall+conv2d_transpose_55/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:Q M
'
_output_shapes
:         d
"
_user_specified_name
input_24
╟!
Ь
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_41194

inputsC
(conv2d_transpose_readvariableop_resource:А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+                           Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╓ 
Ы
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_40228

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╢
F
*__inference_reshape_13_layer_call_fn_40981

inputs
identity╜
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_40405h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └p:P L
(
_output_shapes
:         └p
 
_user_specified_nameinputs
ў
e
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_41099

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:         <<Аh
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:         <<А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         <<А:X T
0
_output_shapes
:         <<А
 
_user_specified_nameinputs
▐ 
Э
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_41089

inputsC
(conv2d_transpose_readvariableop_resource:А@.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype0▌
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ъ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,                           АБ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╥+
В
H__inference_sequential_23_layer_call_and_return_conditional_losses_40449

inputs!
dense_23_40386:	d└p
dense_23_40388:	└p3
conv2d_transpose_52_40407:@@'
conv2d_transpose_52_40409:@4
conv2d_transpose_53_40419:А@(
conv2d_transpose_53_40421:	А5
conv2d_transpose_54_40431:АА(
conv2d_transpose_54_40433:	А4
conv2d_transpose_55_40443:А'
conv2d_transpose_55_40445:
identityИв+conv2d_transpose_52/StatefulPartitionedCallв+conv2d_transpose_53/StatefulPartitionedCallв+conv2d_transpose_54/StatefulPartitionedCallв+conv2d_transpose_55/StatefulPartitionedCallв dense_23/StatefulPartitionedCallє
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_40386dense_23_40388*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └p*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_40385ы
reshape_13/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_40405├
+conv2d_transpose_52/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0conv2d_transpose_52_40407conv2d_transpose_52_40409*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_40228■
leaky_re_lu_69/PartitionedCallPartitionedCall4conv2d_transpose_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_40417╚
+conv2d_transpose_53/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_69/PartitionedCall:output:0conv2d_transpose_53_40419conv2d_transpose_53_40421*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_40272 
leaky_re_lu_70/PartitionedCallPartitionedCall4conv2d_transpose_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <<А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_40429╚
+conv2d_transpose_54/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_70/PartitionedCall:output:0conv2d_transpose_54_40431conv2d_transpose_54_40433*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_40316 
leaky_re_lu_71/PartitionedCallPartitionedCall4conv2d_transpose_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xxА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_40441╟
+conv2d_transpose_55/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0conv2d_transpose_55_40443conv2d_transpose_55_40445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         xx*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_40361Л
IdentityIdentity4conv2d_transpose_55/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         xxб
NoOpNoOp,^conv2d_transpose_52/StatefulPartitionedCall,^conv2d_transpose_53/StatefulPartitionedCall,^conv2d_transpose_54/StatefulPartitionedCall,^conv2d_transpose_55/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         d: : : : : : : : : : 2Z
+conv2d_transpose_52/StatefulPartitionedCall+conv2d_transpose_52/StatefulPartitionedCall2Z
+conv2d_transpose_53/StatefulPartitionedCall+conv2d_transpose_53/StatefulPartitionedCall2Z
+conv2d_transpose_54/StatefulPartitionedCall+conv2d_transpose_54/StatefulPartitionedCall2Z
+conv2d_transpose_55/StatefulPartitionedCall+conv2d_transpose_55/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╟!
Ь
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_40361

inputsC
(conv2d_transpose_readvariableop_resource:А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+                           Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*└
serving_defaultм
=
input_241
serving_default_input_24:0         dO
conv2d_transpose_558
StatefulPartitionedCall:0         xxtensorflow/serving/predict:л╓
╬
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
е
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
 7_jit_compiled_convolution_op"
_tf_keras_layer
е
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
 F_jit_compiled_convolution_op"
_tf_keras_layer
е
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias
 U_jit_compiled_convolution_op"
_tf_keras_layer
f
0
1
&2
'3
54
65
D6
E7
S8
T9"
trackable_list_wrapper
f
0
1
&2
'3
54
65
D6
E7
S8
T9"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
щ
[trace_0
\trace_1
]trace_2
^trace_32■
-__inference_sequential_23_layer_call_fn_40472
-__inference_sequential_23_layer_call_fn_40732
-__inference_sequential_23_layer_call_fn_40757
-__inference_sequential_23_layer_call_fn_40614┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z[trace_0z\trace_1z]trace_2z^trace_3
╒
_trace_0
`trace_1
atrace_2
btrace_32ъ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40857
H__inference_sequential_23_layer_call_and_return_conditional_losses_40957
H__inference_sequential_23_layer_call_and_return_conditional_losses_40647
H__inference_sequential_23_layer_call_and_return_conditional_losses_40680┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z_trace_0z`trace_1zatrace_2zbtrace_3
╠B╔
 __inference__wrapped_model_40191input_24"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
cserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ь
itrace_02╧
(__inference_dense_23_layer_call_fn_40966в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zitrace_0
З
jtrace_02ъ
C__inference_dense_23_layer_call_and_return_conditional_losses_40976в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zjtrace_0
": 	d└p2dense_23/kernel
:└p2dense_23/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
ptrace_02╤
*__inference_reshape_13_layer_call_fn_40981в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zptrace_0
Й
qtrace_02ь
E__inference_reshape_13_layer_call_and_return_conditional_losses_40995в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zqtrace_0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ў
wtrace_02┌
3__inference_conv2d_transpose_52_layer_call_fn_41004в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zwtrace_0
Т
xtrace_02ї
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_41037в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zxtrace_0
4:2@@2conv2d_transpose_52/kernel
&:$@2conv2d_transpose_52/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Є
~trace_02╒
.__inference_leaky_re_lu_69_layer_call_fn_41042в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z~trace_0
Н
trace_02Ё
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_41047в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
∙
Еtrace_02┌
3__inference_conv2d_transpose_53_layer_call_fn_41056в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
Ф
Жtrace_02ї
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_41089в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
5:3А@2conv2d_transpose_53/kernel
':%А2conv2d_transpose_53/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
Ї
Мtrace_02╒
.__inference_leaky_re_lu_70_layer_call_fn_41094в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0
П
Нtrace_02Ё
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_41099в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
∙
Уtrace_02┌
3__inference_conv2d_transpose_54_layer_call_fn_41108в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0
Ф
Фtrace_02ї
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_41141в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
6:4АА2conv2d_transpose_54/kernel
':%А2conv2d_transpose_54/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
Ї
Ъtrace_02╒
.__inference_leaky_re_lu_71_layer_call_fn_41146в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0
П
Ыtrace_02Ё
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_41151в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
∙
бtrace_02┌
3__inference_conv2d_transpose_55_layer_call_fn_41160в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0
Ф
вtrace_02ї
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_41194в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
5:3А2conv2d_transpose_55/kernel
&:$2conv2d_transpose_55/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АB¤
-__inference_sequential_23_layer_call_fn_40472input_24"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
-__inference_sequential_23_layer_call_fn_40732inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
-__inference_sequential_23_layer_call_fn_40757inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
-__inference_sequential_23_layer_call_fn_40614input_24"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЩBЦ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40857inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЩBЦ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40957inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40647input_24"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
H__inference_sequential_23_layer_call_and_return_conditional_losses_40680input_24"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╦B╚
#__inference_signature_wrapper_40707input_24"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_dense_23_layer_call_fn_40966inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_dense_23_layer_call_and_return_conditional_losses_40976inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_reshape_13_layer_call_fn_40981inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_reshape_13_layer_call_and_return_conditional_losses_40995inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
3__inference_conv2d_transpose_52_layer_call_fn_41004inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_41037inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тB▀
.__inference_leaky_re_lu_69_layer_call_fn_41042inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_41047inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
3__inference_conv2d_transpose_53_layer_call_fn_41056inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_41089inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тB▀
.__inference_leaky_re_lu_70_layer_call_fn_41094inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_41099inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
3__inference_conv2d_transpose_54_layer_call_fn_41108inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_41141inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тB▀
.__inference_leaky_re_lu_71_layer_call_fn_41146inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_41151inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
3__inference_conv2d_transpose_55_layer_call_fn_41160inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_41194inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ╖
 __inference__wrapped_model_40191Т
&'56DEST1в.
'в$
"К
input_24         d
к "QкN
L
conv2d_transpose_555К2
conv2d_transpose_55         xxу
N__inference_conv2d_transpose_52_layer_call_and_return_conditional_losses_41037Р&'IвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ ╗
3__inference_conv2d_transpose_52_layer_call_fn_41004Г&'IвF
?в<
:К7
inputs+                           @
к "2К/+                           @ф
N__inference_conv2d_transpose_53_layer_call_and_return_conditional_losses_41089С56IвF
?в<
:К7
inputs+                           @
к "@в=
6К3
0,                           А
Ъ ╝
3__inference_conv2d_transpose_53_layer_call_fn_41056Д56IвF
?в<
:К7
inputs+                           @
к "3К0,                           Ах
N__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_41141ТDEJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
3__inference_conv2d_transpose_54_layer_call_fn_41108ЕDEJвG
@в=
;К8
inputs,                           А
к "3К0,                           Аф
N__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_41194СSTJвG
@в=
;К8
inputs,                           А
к "?в<
5К2
0+                           
Ъ ╝
3__inference_conv2d_transpose_55_layer_call_fn_41160ДSTJвG
@в=
;К8
inputs,                           А
к "2К/+                           д
C__inference_dense_23_layer_call_and_return_conditional_losses_40976]/в,
%в"
 К
inputs         d
к "&в#
К
0         └p
Ъ |
(__inference_dense_23_layer_call_fn_40966P/в,
%в"
 К
inputs         d
к "К         └p╡
I__inference_leaky_re_lu_69_layer_call_and_return_conditional_losses_41047h7в4
-в*
(К%
inputs         @
к "-в*
#К 
0         @
Ъ Н
.__inference_leaky_re_lu_69_layer_call_fn_41042[7в4
-в*
(К%
inputs         @
к " К         @╖
I__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_41099j8в5
.в+
)К&
inputs         <<А
к ".в+
$К!
0         <<А
Ъ П
.__inference_leaky_re_lu_70_layer_call_fn_41094]8в5
.в+
)К&
inputs         <<А
к "!К         <<А╖
I__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_41151j8в5
.в+
)К&
inputs         xxА
к ".в+
$К!
0         xxА
Ъ П
.__inference_leaky_re_lu_71_layer_call_fn_41146]8в5
.в+
)К&
inputs         xxА
к "!К         xxАк
E__inference_reshape_13_layer_call_and_return_conditional_losses_40995a0в-
&в#
!К
inputs         └p
к "-в*
#К 
0         @
Ъ В
*__inference_reshape_13_layer_call_fn_40981T0в-
&в#
!К
inputs         └p
к " К         @┬
H__inference_sequential_23_layer_call_and_return_conditional_losses_40647v
&'56DEST9в6
/в,
"К
input_24         d
p 

 
к "-в*
#К 
0         xx
Ъ ┬
H__inference_sequential_23_layer_call_and_return_conditional_losses_40680v
&'56DEST9в6
/в,
"К
input_24         d
p

 
к "-в*
#К 
0         xx
Ъ └
H__inference_sequential_23_layer_call_and_return_conditional_losses_40857t
&'56DEST7в4
-в*
 К
inputs         d
p 

 
к "-в*
#К 
0         xx
Ъ └
H__inference_sequential_23_layer_call_and_return_conditional_losses_40957t
&'56DEST7в4
-в*
 К
inputs         d
p

 
к "-в*
#К 
0         xx
Ъ Ъ
-__inference_sequential_23_layer_call_fn_40472i
&'56DEST9в6
/в,
"К
input_24         d
p 

 
к " К         xxЪ
-__inference_sequential_23_layer_call_fn_40614i
&'56DEST9в6
/в,
"К
input_24         d
p

 
к " К         xxШ
-__inference_sequential_23_layer_call_fn_40732g
&'56DEST7в4
-в*
 К
inputs         d
p 

 
к " К         xxШ
-__inference_sequential_23_layer_call_fn_40757g
&'56DEST7в4
-в*
 К
inputs         d
p

 
к " К         xx╞
#__inference_signature_wrapper_40707Ю
&'56DEST=в:
в 
3к0
.
input_24"К
input_24         d"QкN
L
conv2d_transpose_555К2
conv2d_transpose_55         xx