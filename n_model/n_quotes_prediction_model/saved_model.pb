??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
S
	Bucketize

input"T

output"
Ttype:
2	"

boundarieslist(float)
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
sequential_3/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namesequential_3/dense_12/kernel
?
0sequential_3/dense_12/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_12/kernel* 
_output_shapes
:
??*
dtype0
?
sequential_3/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namesequential_3/dense_12/bias
?
.sequential_3/dense_12/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_12/bias*
_output_shapes	
:?*
dtype0
?
sequential_3/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namesequential_3/dense_13/kernel
?
0sequential_3/dense_13/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_13/kernel* 
_output_shapes
:
??*
dtype0
?
sequential_3/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namesequential_3/dense_13/bias
?
.sequential_3/dense_13/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_13/bias*
_output_shapes	
:?*
dtype0
?
sequential_3/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namesequential_3/dense_14/kernel
?
0sequential_3/dense_14/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_14/kernel* 
_output_shapes
:
??*
dtype0
?
sequential_3/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namesequential_3/dense_14/bias
?
.sequential_3/dense_14/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_14/bias*
_output_shapes	
:?*
dtype0
?
sequential_3/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_namesequential_3/dense_15/kernel
?
0sequential_3/dense_15/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_15/kernel*
_output_shapes
:	?*
dtype0
?
sequential_3/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesequential_3/dense_15/bias
?
.sequential_3/dense_15/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_15/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name126*
value_dtype0	
m
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name162*
value_dtype0	
m
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name198*
value_dtype0	
m
hash_table_3HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name234*
value_dtype0	
m
hash_table_4HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name270*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
#Adam/sequential_3/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adam/sequential_3/dense_12/kernel/m
?
7Adam/sequential_3/dense_12/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/dense_12/kernel/m* 
_output_shapes
:
??*
dtype0
?
!Adam/sequential_3/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_3/dense_12/bias/m
?
5Adam/sequential_3/dense_12/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/dense_12/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_3/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adam/sequential_3/dense_13/kernel/m
?
7Adam/sequential_3/dense_13/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/dense_13/kernel/m* 
_output_shapes
:
??*
dtype0
?
!Adam/sequential_3/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_3/dense_13/bias/m
?
5Adam/sequential_3/dense_13/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/dense_13/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_3/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adam/sequential_3/dense_14/kernel/m
?
7Adam/sequential_3/dense_14/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/dense_14/kernel/m* 
_output_shapes
:
??*
dtype0
?
!Adam/sequential_3/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_3/dense_14/bias/m
?
5Adam/sequential_3/dense_14/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/dense_14/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_3/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#Adam/sequential_3/dense_15/kernel/m
?
7Adam/sequential_3/dense_15/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/dense_15/kernel/m*
_output_shapes
:	?*
dtype0
?
!Adam/sequential_3/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_3/dense_15/bias/m
?
5Adam/sequential_3/dense_15/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/dense_15/bias/m*
_output_shapes
:*
dtype0
?
#Adam/sequential_3/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adam/sequential_3/dense_12/kernel/v
?
7Adam/sequential_3/dense_12/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/dense_12/kernel/v* 
_output_shapes
:
??*
dtype0
?
!Adam/sequential_3/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_3/dense_12/bias/v
?
5Adam/sequential_3/dense_12/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/dense_12/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_3/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adam/sequential_3/dense_13/kernel/v
?
7Adam/sequential_3/dense_13/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/dense_13/kernel/v* 
_output_shapes
:
??*
dtype0
?
!Adam/sequential_3/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_3/dense_13/bias/v
?
5Adam/sequential_3/dense_13/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/dense_13/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_3/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adam/sequential_3/dense_14/kernel/v
?
7Adam/sequential_3/dense_14/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/dense_14/kernel/v* 
_output_shapes
:
??*
dtype0
?
!Adam/sequential_3/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_3/dense_14/bias/v
?
5Adam/sequential_3/dense_14/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/dense_14/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_3/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#Adam/sequential_3/dense_15/kernel/v
?
7Adam/sequential_3/dense_15/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/dense_15/kernel/v*
_output_shapes
:	?*
dtype0
?
!Adam/sequential_3/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_3/dense_15/bias/v
?
5Adam/sequential_3/dense_15/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/dense_15/bias/v*
_output_shapes
:*
dtype0
P
ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_4Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
t
Const_5Const*
_output_shapes
:*
dtype0*9
value0B.BjoyBangerBfearBneutralBsadBcalm
?
Const_6Const*
_output_shapes
:*
dtype0	*E
value<B:	"0                                           
c
Const_7Const*
_output_shapes
:*
dtype0*(
valueBBfemaleBmaleBother
h
Const_8Const*
_output_shapes
:*
dtype0	*-
value$B"	"                      
?#
Const_9Const*
_output_shapes	
:?*
dtype0*?#
value?#B?#?Bsouth africaBspainBunited states of americaB	australiaBzimbabweBunited kingdomBcolombiaBuruguayBfranceBcanadaBpolandBgermanyBhungaryBindiaBphilippinesBnamibiaB	argentinaBnigeriaBirelandBnorwayBsoviet unionBpeople's republic of chinaBnew zealandB
bangladeshBromaniaBswedenBmalaysiaBiranBaustriaBrwandaBsyriaB	singaporeBpakistanBdenmarkBghanaBenglandBbelgiumBkingdom of the netherlandsBportugalBmexicoBjapanB	indonesiaBisraelBbrazilBcubaBczechoslovakiaBtaiwanBrussiaBkosovoByemenBparaguayB	venezuelaBbarbadosBitalyBmoroccoBstate of palestineBbritish indiaBkenyaBzambiaBfinlandB	guatemalaBiraqBjamaicaBthe bahamasBswitzerlandBsouth koreaBugandaB	hong kongBtanzaniaB	canadiansBestoniaB	sri lankaBbotswanaBturkeyBmyanmarBmalawiB
mauritaniaBgreeceBslovakiaBguyanaBnorth macedoniaBcambodiaBchileB
costa ricaBukraineBbhutanBsolomon islandsBlebanonBafghanistanBkuwaitBcôte d'ivoireBnepalBpapua new guineaBmoldovaBegyptBtunisiaBbulgariaBserbiaBczech republicB
uzbekistanBbosnia and herzegovinaBbahrainBmaltaBicelandB	nicaraguaBalgeriaBnorth koreaBperuBjordanBpuerto ricoBdominican republicBsenegalBczechoslovak republicB
yugoslaviaBdominicaBtrinidad and tobagoB democratic republic of the congoBpanamaBeswatiniBsloveniaBsierra leoneBmaldivesB
seychellesBqatarBwalesBlatviaBangolaBthailandBtongaBcroatiaBunited arab emiratesB(socialist federal republic of yugoslaviaBmongoliaBgeorgiaBecuadorBalbaniaBscotlandBsaudi arabiaB
kazakhstanBbritish hong kongBsomaliaB	mauritiusBnigerBgerman democratic republicBvietnamB
azerbaijanBhondurasB	lithuaniaBbelizeBeritreaBantigua and barbudaBrepublic of chinaBsaint luciaBcentral african republicBnorthern irelandB
the gambiaBsecond syrian republicBsamoaBsurinameBfijiBboliviaBsudanBcameroonBgabonB
luxembourgBarmeniaBkingdom of bulgariaBliechtensteinBamerican samoaBcyprusBmaliBtogoBbeninBel salvadorBhaitiBchinaBethiopiaB
madagascarBgrenadaBliberiaBvanuatuB saint vincent and the grenadinesBguernseyBkiribatiB	palestineBlibyaBgerman empireBindiansB
kyrgyzstanB
somalilandBsecond polish republicBbelarusBba'athist iraqB	gibraltarBcanadian frenchBbermudaB#turkish republic of northern cyprusBpitcairn islandsBdjiboutiBguineaBfederal republic of yugoslaviaB+united kingdom of great britain and irelandB
mozambiqueBtimor-lesteBsouth sudanBisle of manBlesothoBomanBburundiBlaosBfaroe islandsBandorraB	greenlandBcayman islandsBbritish national (overseas)B
cape verdeBfree city of danzigBmalayaBpahlavi dynastyBmonacoBburkina fasoBguinea-bissauBkingdom of romaniaBcomorosBcanada–united states borderBvitacuraBunited states virgin islandsBsantiago de cuba provinceBrhodesiaBaustraliansBczechoslovak socialist republicBrepublic of the congoBbruneiBtuvaluBbyblosBsão tomé and príncipeB#protectorate of bohemia and moraviaBchadBsaint kitts and nevisBcook islandsBwest germanyB
san marinoBkingdom of italyBnazi germanyBpalauBguamB
montenegroB"colony and protectorate of nigeriaBfalkland islandsB#ukrainian soviet socialist republicBmarshall islandsBunion of south africaB
tajikistanB'federal people's republic of yugoslaviaBpalestiniansBcuraçaoB
quebeckersBgerman reichBdanish realmBpolesBfirst syrian republicBequatorial guineaB	americansB,russian socialist federative soviet republicBorganizationBnauruBglaspyB	kurdistanBkingdom of greeceBfederated states of micronesiaBunited arab republicBmexico–united states borderBaustria-hungaryBrepublic of abkhaziaBzaireBkingdom of egyptBkingdom of iraqBpalestinian national authorityBsouth vietnamBmacauBsouthern rhodesiaBmongolian people's republicBkenya colonyBcongoBtunisBrepublic of veniceBtibetBbritish virgin islandsBthe netherlandsBkingdom of hungaryBgermanB
montserratB
makhmalbafBgerman wikipediaBkingdom of sikkimBpolish people's republicB
malaysiansBderryBirish peopleBnorfolk islandBenglish peopleBmandatory palestineBmexico cityBweimar republicBperuanaBmexicaliBanguillaB sahrawi arab democratic republicBsyrianBfrenchB
pakistanisBnew zealandersB&people's socialist republic of albaniaBenglish wikipediaBvatican cityBamericanBarubaBafricaBalawite stateBkingdom of englandBrussian empireBkingdom of afghanistanB$federation of rhodesia and nyasalandBtemucoBunified teamBunkBtaiwan under japanese ruleBserbia and montenegroBnorthern mariana islands
?
Const_10Const*
_output_shapes	
:?*
dtype0	*?
value?B?	?"?                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      
?
Const_11Const*
_output_shapes
:*
dtype0*i
value`B^B
politicianBathleteBmusicianB
researcherBlawyerB
journalistBactorBbusinessperson
?
Const_12Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@                                                         
?
Const_13Const*
_output_shapes
:*
dtype0*N
valueEBCBartBsportBeconomy & financeBpoliticsBhealth & science
y
Const_14Const*
_output_shapes
:*
dtype0	*=
value4B2	"(                                    
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_5Const_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_466642
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_7Const_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_466650
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_2Const_9Const_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_466658
?
StatefulPartitionedCall_3StatefulPartitionedCallhash_table_3Const_11Const_12*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_466666
?
StatefulPartitionedCall_4StatefulPartitionedCallhash_table_4Const_13Const_14*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_466674
?
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4
?5
Const_15Const"/device:CPU:0*
_output_shapes
: *
dtype0*?5
value?5B?5 B?5
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
_build_input_shape
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
x
_feature_columns

_resources
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?
+iter

,beta_1

-beta_2
	.decay
/learning_ratemhmimjmkml mm%mn&movpvqvrvsvt vu%vv&vw
 
8
0
1
2
3
4
 5
%6
&7
8
0
1
2
3
4
 5
%6
&7
 
?
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
	trainable_variables

regularization_losses
 
 
E
5emotion

6gender
7nationality
8
occupation
	9topic
 
 
 
?
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
hf
VARIABLE_VALUEsequential_3/dense_12/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_3/dense_12/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
hf
VARIABLE_VALUEsequential_3/dense_13/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_3/dense_13/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
hf
VARIABLE_VALUEsequential_3/dense_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_3/dense_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
!	variables
"trainable_variables
#regularization_losses
hf
VARIABLE_VALUEsequential_3/dense_15/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_3/dense_15/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
'	variables
(trainable_variables
)regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

S0
T1
 
 

Uemotion_lookup

Vgender_lookup

Wnationality_lookup

Xoccupation_lookup

Ytopic_lookup
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ztotal
	[count
\	variables
]	keras_api
D
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api

c_initializer

d_initializer

e_initializer

f_initializer

g_initializer
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

\	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

a	variables
 
 
 
 
 
??
VARIABLE_VALUE#Adam/sequential_3/dense_12/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_3/dense_12/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_3/dense_13/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_3/dense_13/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_3/dense_14/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_3/dense_14/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_3/dense_15/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_3/dense_15/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_3/dense_12/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_3/dense_12/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_3/dense_13/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_3/dense_13/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_3/dense_14/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_3/dense_14/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_3/dense_15/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_3/dense_15/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
n
serving_default_agePlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
r
serving_default_emotionPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
q
serving_default_genderPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_nationalityPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_occupationPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
p
serving_default_topicPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_5StatefulPartitionedCallserving_default_ageserving_default_emotionserving_default_genderserving_default_nationalityserving_default_occupationserving_default_topic
hash_tableConsthash_table_1Const_1hash_table_2Const_2hash_table_3Const_3hash_table_4Const_4sequential_3/dense_12/kernelsequential_3/dense_12/biassequential_3/dense_13/kernelsequential_3/dense_13/biassequential_3/dense_14/kernelsequential_3/dense_14/biassequential_3/dense_15/kernelsequential_3/dense_15/bias*#
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_465624
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_6StatefulPartitionedCallsaver_filename0sequential_3/dense_12/kernel/Read/ReadVariableOp.sequential_3/dense_12/bias/Read/ReadVariableOp0sequential_3/dense_13/kernel/Read/ReadVariableOp.sequential_3/dense_13/bias/Read/ReadVariableOp0sequential_3/dense_14/kernel/Read/ReadVariableOp.sequential_3/dense_14/bias/Read/ReadVariableOp0sequential_3/dense_15/kernel/Read/ReadVariableOp.sequential_3/dense_15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/sequential_3/dense_12/kernel/m/Read/ReadVariableOp5Adam/sequential_3/dense_12/bias/m/Read/ReadVariableOp7Adam/sequential_3/dense_13/kernel/m/Read/ReadVariableOp5Adam/sequential_3/dense_13/bias/m/Read/ReadVariableOp7Adam/sequential_3/dense_14/kernel/m/Read/ReadVariableOp5Adam/sequential_3/dense_14/bias/m/Read/ReadVariableOp7Adam/sequential_3/dense_15/kernel/m/Read/ReadVariableOp5Adam/sequential_3/dense_15/bias/m/Read/ReadVariableOp7Adam/sequential_3/dense_12/kernel/v/Read/ReadVariableOp5Adam/sequential_3/dense_12/bias/v/Read/ReadVariableOp7Adam/sequential_3/dense_13/kernel/v/Read/ReadVariableOp5Adam/sequential_3/dense_13/bias/v/Read/ReadVariableOp7Adam/sequential_3/dense_14/kernel/v/Read/ReadVariableOp5Adam/sequential_3/dense_14/bias/v/Read/ReadVariableOp7Adam/sequential_3/dense_15/kernel/v/Read/ReadVariableOp5Adam/sequential_3/dense_15/bias/v/Read/ReadVariableOpConst_15*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_466826
?	
StatefulPartitionedCall_7StatefulPartitionedCallsaver_filenamesequential_3/dense_12/kernelsequential_3/dense_12/biassequential_3/dense_13/kernelsequential_3/dense_13/biassequential_3/dense_14/kernelsequential_3/dense_14/biassequential_3/dense_15/kernelsequential_3/dense_15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1#Adam/sequential_3/dense_12/kernel/m!Adam/sequential_3/dense_12/bias/m#Adam/sequential_3/dense_13/kernel/m!Adam/sequential_3/dense_13/bias/m#Adam/sequential_3/dense_14/kernel/m!Adam/sequential_3/dense_14/bias/m#Adam/sequential_3/dense_15/kernel/m!Adam/sequential_3/dense_15/bias/m#Adam/sequential_3/dense_12/kernel/v!Adam/sequential_3/dense_12/bias/v#Adam/sequential_3/dense_13/kernel/v!Adam/sequential_3/dense_13/bias/v#Adam/sequential_3/dense_14/kernel/v!Adam/sequential_3/dense_14/bias/v#Adam/sequential_3/dense_15/kernel/v!Adam/sequential_3/dense_15/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_466935??
?
?
__inference_<lambda>_4666422
.table_init125_lookuptableimportv2_table_handle*
&table_init125_lookuptableimportv2_keys,
(table_init125_lookuptableimportv2_values	
identity??!table_init125/LookupTableImportV2?
!table_init125/LookupTableImportV2LookupTableImportV2.table_init125_lookuptableimportv2_table_handle&table_init125_lookuptableimportv2_keys(table_init125_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init125/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init125/LookupTableImportV2!table_init125/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?%
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_465520
age	
emotion

gender
nationality

occupation	
topic
dense_features_465478
dense_features_465480	
dense_features_465482
dense_features_465484	
dense_features_465486
dense_features_465488	
dense_features_465490
dense_features_465492	
dense_features_465494
dense_features_465496	#
dense_12_465499:
??
dense_12_465501:	?#
dense_13_465504:
??
dense_13_465506:	?#
dense_14_465509:
??
dense_14_465511:	?"
dense_15_465514:	?
dense_15_465516:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallageemotiongendernationality
occupationtopicdense_features_465478dense_features_465480dense_features_465482dense_features_465484dense_features_465486dense_features_465488dense_features_465490dense_features_465492dense_features_465494dense_features_465496*
Tin
2						*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_464893?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_12_465499dense_12_465501*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_464926?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_465504dense_13_465506*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_464943?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_465509dense_14_465511*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_464960?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_465514dense_15_465516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_464977x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:H D
#
_output_shapes
:?????????

_user_specified_nameage:LH
#
_output_shapes
:?????????
!
_user_specified_name	emotion:KG
#
_output_shapes
:?????????
 
_user_specified_namegender:PL
#
_output_shapes
:?????????
%
_user_specified_namenationality:OK
#
_output_shapes
:?????????
$
_user_specified_name
occupation:JF
#
_output_shapes
:?????????

_user_specified_nametopic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
;
__inference__creator_466621
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name270*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
-__inference_sequential_3_layer_call_fn_465716

inputs_age	
inputs_emotion
inputs_gender
inputs_nationality
inputs_occupation
inputs_topic
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9:
??

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_ageinputs_emotioninputs_genderinputs_nationalityinputs_occupationinputs_topicunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*#
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_465385o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/age:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/emotion:RN
#
_output_shapes
:?????????
'
_user_specified_nameinputs/gender:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/nationality:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/topic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference__initializer_4666112
.table_init233_lookuptableimportv2_table_handle*
&table_init233_lookuptableimportv2_keys,
(table_init233_lookuptableimportv2_values	
identity??!table_init233/LookupTableImportV2?
!table_init233/LookupTableImportV2LookupTableImportV2.table_init233_lookuptableimportv2_table_handle&table_init233_lookuptableimportv2_keys(table_init233_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init233/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init233/LookupTableImportV2!table_init233/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference__initializer_4665752
.table_init161_lookuptableimportv2_table_handle*
&table_init161_lookuptableimportv2_keys,
(table_init161_lookuptableimportv2_values	
identity??!table_init161/LookupTableImportV2?
!table_init161/LookupTableImportV2LookupTableImportV2.table_init161_lookuptableimportv2_table_handle&table_init161_lookuptableimportv2_keys(table_init161_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init161/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init161/LookupTableImportV2!table_init161/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference_<lambda>_4666662
.table_init233_lookuptableimportv2_table_handle*
&table_init233_lookuptableimportv2_keys,
(table_init233_lookuptableimportv2_values	
identity??!table_init233/LookupTableImportV2?
!table_init233/LookupTableImportV2LookupTableImportV2.table_init233_lookuptableimportv2_table_handle&table_init233_lookuptableimportv2_keys(table_init233_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init233/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init233/LookupTableImportV2!table_init233/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
-__inference_sequential_3_layer_call_fn_465023
age	
emotion

gender
nationality

occupation	
topic
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9:
??

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallageemotiongendernationality
occupationtopicunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*#
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_464984o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:?????????

_user_specified_nameage:LH
#
_output_shapes
:?????????
!
_user_specified_name	emotion:KG
#
_output_shapes
:?????????
 
_user_specified_namegender:PL
#
_output_shapes
:?????????
%
_user_specified_namenationality:OK
#
_output_shapes
:?????????
$
_user_specified_name
occupation:JF
#
_output_shapes
:?????????

_user_specified_nametopic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_dense_15_layer_call_fn_466533

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_464977o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_12_layer_call_and_return_conditional_losses_464926

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
;
__inference__creator_466603
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name234*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference__initializer_4665572
.table_init125_lookuptableimportv2_table_handle*
&table_init125_lookuptableimportv2_keys,
(table_init125_lookuptableimportv2_values	
identity??!table_init125/LookupTableImportV2?
!table_init125/LookupTableImportV2LookupTableImportV2.table_init125_lookuptableimportv2_table_handle&table_init125_lookuptableimportv2_keys(table_init125_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init125/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init125/LookupTableImportV2!table_init125/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
;
__inference__creator_466585
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name198*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
-
__inference__destroyer_466616
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_4666292
.table_init269_lookuptableimportv2_table_handle*
&table_init269_lookuptableimportv2_keys,
(table_init269_lookuptableimportv2_values	
identity??!table_init269/LookupTableImportV2?
!table_init269/LookupTableImportV2LookupTableImportV2.table_init269_lookuptableimportv2_table_handle&table_init269_lookuptableimportv2_keys(table_init269_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init269/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init269/LookupTableImportV2!table_init269/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?K
?
__inference__traced_save_466826
file_prefix;
7savev2_sequential_3_dense_12_kernel_read_readvariableop9
5savev2_sequential_3_dense_12_bias_read_readvariableop;
7savev2_sequential_3_dense_13_kernel_read_readvariableop9
5savev2_sequential_3_dense_13_bias_read_readvariableop;
7savev2_sequential_3_dense_14_kernel_read_readvariableop9
5savev2_sequential_3_dense_14_bias_read_readvariableop;
7savev2_sequential_3_dense_15_kernel_read_readvariableop9
5savev2_sequential_3_dense_15_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adam_sequential_3_dense_12_kernel_m_read_readvariableop@
<savev2_adam_sequential_3_dense_12_bias_m_read_readvariableopB
>savev2_adam_sequential_3_dense_13_kernel_m_read_readvariableop@
<savev2_adam_sequential_3_dense_13_bias_m_read_readvariableopB
>savev2_adam_sequential_3_dense_14_kernel_m_read_readvariableop@
<savev2_adam_sequential_3_dense_14_bias_m_read_readvariableopB
>savev2_adam_sequential_3_dense_15_kernel_m_read_readvariableop@
<savev2_adam_sequential_3_dense_15_bias_m_read_readvariableopB
>savev2_adam_sequential_3_dense_12_kernel_v_read_readvariableop@
<savev2_adam_sequential_3_dense_12_bias_v_read_readvariableopB
>savev2_adam_sequential_3_dense_13_kernel_v_read_readvariableop@
<savev2_adam_sequential_3_dense_13_bias_v_read_readvariableopB
>savev2_adam_sequential_3_dense_14_kernel_v_read_readvariableop@
<savev2_adam_sequential_3_dense_14_bias_v_read_readvariableopB
>savev2_adam_sequential_3_dense_15_kernel_v_read_readvariableop@
<savev2_adam_sequential_3_dense_15_bias_v_read_readvariableop
savev2_const_15

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_sequential_3_dense_12_kernel_read_readvariableop5savev2_sequential_3_dense_12_bias_read_readvariableop7savev2_sequential_3_dense_13_kernel_read_readvariableop5savev2_sequential_3_dense_13_bias_read_readvariableop7savev2_sequential_3_dense_14_kernel_read_readvariableop5savev2_sequential_3_dense_14_bias_read_readvariableop7savev2_sequential_3_dense_15_kernel_read_readvariableop5savev2_sequential_3_dense_15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_sequential_3_dense_12_kernel_m_read_readvariableop<savev2_adam_sequential_3_dense_12_bias_m_read_readvariableop>savev2_adam_sequential_3_dense_13_kernel_m_read_readvariableop<savev2_adam_sequential_3_dense_13_bias_m_read_readvariableop>savev2_adam_sequential_3_dense_14_kernel_m_read_readvariableop<savev2_adam_sequential_3_dense_14_bias_m_read_readvariableop>savev2_adam_sequential_3_dense_15_kernel_m_read_readvariableop<savev2_adam_sequential_3_dense_15_bias_m_read_readvariableop>savev2_adam_sequential_3_dense_12_kernel_v_read_readvariableop<savev2_adam_sequential_3_dense_12_bias_v_read_readvariableop>savev2_adam_sequential_3_dense_13_kernel_v_read_readvariableop<savev2_adam_sequential_3_dense_13_bias_v_read_readvariableop>savev2_adam_sequential_3_dense_14_kernel_v_read_readvariableop<savev2_adam_sequential_3_dense_14_bias_v_read_readvariableop>savev2_adam_sequential_3_dense_15_kernel_v_read_readvariableop<savev2_adam_sequential_3_dense_15_bias_v_read_readvariableopsavev2_const_15"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:
??:?:	?:: : : : : : : : : :
??:?:
??:?:
??:?:	?::
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::"

_output_shapes
: 
??
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_465902

inputs_age	
inputs_emotion
inputs_gender
inputs_nationality
inputs_occupation
inputs_topicU
Qdense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_table_handleV
Rdense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_default_value	T
Pdense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_table_handleU
Qdense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_default_value	Y
Udense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleZ
Vdense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_default_value	X
Tdense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_table_handleY
Udense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_default_value	S
Odense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_table_handleT
Pdense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_default_value	;
'dense_12_matmul_readvariableop_resource:
??7
(dense_12_biasadd_readvariableop_resource:	?;
'dense_13_matmul_readvariableop_resource:
??7
(dense_13_biasadd_readvariableop_resource:	?;
'dense_14_matmul_readvariableop_resource:
??7
(dense_14_biasadd_readvariableop_resource:	?:
'dense_15_matmul_readvariableop_resource:	?6
(dense_15_biasadd_readvariableop_resource:
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?Ddense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2?Cdense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2?Hdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2?Gdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2?Bdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2w
,dense_features/age_bucketized/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(dense_features/age_bucketized/ExpandDims
ExpandDims
inputs_age5dense_features/age_bucketized/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
"dense_features/age_bucketized/CastCast1dense_features/age_bucketized/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
'dense_features/age_bucketized/Bucketize	Bucketize&dense_features/age_bucketized/Cast:y:0*
T0*'
_output_shapes
:?????????*^

boundariesP
N"L  ?@   A  pA  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B  ?B  ?B  ?B  ?B  ?B  ?B?
$dense_features/age_bucketized/Cast_1Cast0dense_features/age_bucketized/Bucketize:output:0*

DstT0	*

SrcT0*'
_output_shapes
:?????????p
+dense_features/age_bucketized/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??r
-dense_features/age_bucketized/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    m
+dense_features/age_bucketized/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
%dense_features/age_bucketized/one_hotOneHot(dense_features/age_bucketized/Cast_1:y:04dense_features/age_bucketized/one_hot/depth:output:04dense_features/age_bucketized/one_hot/Const:output:06dense_features/age_bucketized/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
#dense_features/age_bucketized/ShapeShape.dense_features/age_bucketized/one_hot:output:0*
T0*
_output_shapes
:{
1dense_features/age_bucketized/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_features/age_bucketized/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_features/age_bucketized/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_features/age_bucketized/strided_sliceStridedSlice,dense_features/age_bucketized/Shape:output:0:dense_features/age_bucketized/strided_slice/stack:output:0<dense_features/age_bucketized/strided_slice/stack_1:output:0<dense_features/age_bucketized/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-dense_features/age_bucketized/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/age_bucketized/Reshape/shapePack4dense_features/age_bucketized/strided_slice:output:06dense_features/age_bucketized/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
%dense_features/age_bucketized/ReshapeReshape.dense_features/age_bucketized/one_hot:output:04dense_features/age_bucketized/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/dense_features/emotion_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+dense_features/emotion_indicator/ExpandDims
ExpandDimsinputs_emotion8dense_features/emotion_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
?dense_features/emotion_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
9dense_features/emotion_indicator/to_sparse_input/NotEqualNotEqual4dense_features/emotion_indicator/ExpandDims:output:0Hdense_features/emotion_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
8dense_features/emotion_indicator/to_sparse_input/indicesWhere=dense_features/emotion_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
7dense_features/emotion_indicator/to_sparse_input/valuesGatherNd4dense_features/emotion_indicator/ExpandDims:output:0@dense_features/emotion_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
<dense_features/emotion_indicator/to_sparse_input/dense_shapeShape4dense_features/emotion_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Ddense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qdense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_table_handle@dense_features/emotion_indicator/to_sparse_input/values:output:0Rdense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
<dense_features/emotion_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
.dense_features/emotion_indicator/SparseToDenseSparseToDense@dense_features/emotion_indicator/to_sparse_input/indices:index:0Edense_features/emotion_indicator/to_sparse_input/dense_shape:output:0Mdense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2:values:0Edense_features/emotion_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????s
.dense_features/emotion_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
0dense_features/emotion_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    p
.dense_features/emotion_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
(dense_features/emotion_indicator/one_hotOneHot6dense_features/emotion_indicator/SparseToDense:dense:07dense_features/emotion_indicator/one_hot/depth:output:07dense_features/emotion_indicator/one_hot/Const:output:09dense_features/emotion_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
6dense_features/emotion_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
$dense_features/emotion_indicator/SumSum1dense_features/emotion_indicator/one_hot:output:0?dense_features/emotion_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
&dense_features/emotion_indicator/ShapeShape-dense_features/emotion_indicator/Sum:output:0*
T0*
_output_shapes
:~
4dense_features/emotion_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/emotion_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/emotion_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/emotion_indicator/strided_sliceStridedSlice/dense_features/emotion_indicator/Shape:output:0=dense_features/emotion_indicator/strided_slice/stack:output:0?dense_features/emotion_indicator/strided_slice/stack_1:output:0?dense_features/emotion_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/emotion_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/emotion_indicator/Reshape/shapePack7dense_features/emotion_indicator/strided_slice:output:09dense_features/emotion_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/emotion_indicator/ReshapeReshape-dense_features/emotion_indicator/Sum:output:07dense_features/emotion_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
.dense_features/gender_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
*dense_features/gender_indicator/ExpandDims
ExpandDimsinputs_gender7dense_features/gender_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????
>dense_features/gender_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
8dense_features/gender_indicator/to_sparse_input/NotEqualNotEqual3dense_features/gender_indicator/ExpandDims:output:0Gdense_features/gender_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
7dense_features/gender_indicator/to_sparse_input/indicesWhere<dense_features/gender_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
6dense_features/gender_indicator/to_sparse_input/valuesGatherNd3dense_features/gender_indicator/ExpandDims:output:0?dense_features/gender_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
;dense_features/gender_indicator/to_sparse_input/dense_shapeShape3dense_features/gender_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Cdense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Pdense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_table_handle?dense_features/gender_indicator/to_sparse_input/values:output:0Qdense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
;dense_features/gender_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
-dense_features/gender_indicator/SparseToDenseSparseToDense?dense_features/gender_indicator/to_sparse_input/indices:index:0Ddense_features/gender_indicator/to_sparse_input/dense_shape:output:0Ldense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2:values:0Ddense_features/gender_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????r
-dense_features/gender_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
/dense_features/gender_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    o
-dense_features/gender_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
'dense_features/gender_indicator/one_hotOneHot5dense_features/gender_indicator/SparseToDense:dense:06dense_features/gender_indicator/one_hot/depth:output:06dense_features/gender_indicator/one_hot/Const:output:08dense_features/gender_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
5dense_features/gender_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
#dense_features/gender_indicator/SumSum0dense_features/gender_indicator/one_hot:output:0>dense_features/gender_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
%dense_features/gender_indicator/ShapeShape,dense_features/gender_indicator/Sum:output:0*
T0*
_output_shapes
:}
3dense_features/gender_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/gender_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/gender_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_features/gender_indicator/strided_sliceStridedSlice.dense_features/gender_indicator/Shape:output:0<dense_features/gender_indicator/strided_slice/stack:output:0>dense_features/gender_indicator/strided_slice/stack_1:output:0>dense_features/gender_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/gender_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
-dense_features/gender_indicator/Reshape/shapePack6dense_features/gender_indicator/strided_slice:output:08dense_features/gender_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
'dense_features/gender_indicator/ReshapeReshape,dense_features/gender_indicator/Sum:output:06dense_features/gender_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3dense_features/nationality_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/dense_features/nationality_indicator/ExpandDims
ExpandDimsinputs_nationality<dense_features/nationality_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/nationality_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/nationality_indicator/to_sparse_input/NotEqualNotEqual8dense_features/nationality_indicator/ExpandDims:output:0Ldense_features/nationality_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/nationality_indicator/to_sparse_input/indicesWhereAdense_features/nationality_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/nationality_indicator/to_sparse_input/valuesGatherNd8dense_features/nationality_indicator/ExpandDims:output:0Ddense_features/nationality_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/nationality_indicator/to_sparse_input/dense_shapeShape8dense_features/nationality_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Hdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Udense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleDdense_features/nationality_indicator/to_sparse_input/values:output:0Vdense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/nationality_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/nationality_indicator/SparseToDenseSparseToDenseDdense_features/nationality_indicator/to_sparse_input/indices:index:0Idense_features/nationality_indicator/to_sparse_input/dense_shape:output:0Qdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2:values:0Idense_features/nationality_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/nationality_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/nationality_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    u
2dense_features/nationality_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
,dense_features/nationality_indicator/one_hotOneHot:dense_features/nationality_indicator/SparseToDense:dense:0;dense_features/nationality_indicator/one_hot/depth:output:0;dense_features/nationality_indicator/one_hot/Const:output:0=dense_features/nationality_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
:dense_features/nationality_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/nationality_indicator/SumSum5dense_features/nationality_indicator/one_hot:output:0Cdense_features/nationality_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
*dense_features/nationality_indicator/ShapeShape1dense_features/nationality_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/nationality_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/nationality_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/nationality_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/nationality_indicator/strided_sliceStridedSlice3dense_features/nationality_indicator/Shape:output:0Adense_features/nationality_indicator/strided_slice/stack:output:0Cdense_features/nationality_indicator/strided_slice/stack_1:output:0Cdense_features/nationality_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
4dense_features/nationality_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
2dense_features/nationality_indicator/Reshape/shapePack;dense_features/nationality_indicator/strided_slice:output:0=dense_features/nationality_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/nationality_indicator/ReshapeReshape1dense_features/nationality_indicator/Sum:output:0;dense_features/nationality_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????}
2dense_features/occupation_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/occupation_indicator/ExpandDims
ExpandDimsinputs_occupation;dense_features/occupation_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/occupation_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/occupation_indicator/to_sparse_input/NotEqualNotEqual7dense_features/occupation_indicator/ExpandDims:output:0Kdense_features/occupation_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/occupation_indicator/to_sparse_input/indicesWhere@dense_features/occupation_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/occupation_indicator/to_sparse_input/valuesGatherNd7dense_features/occupation_indicator/ExpandDims:output:0Cdense_features/occupation_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/occupation_indicator/to_sparse_input/dense_shapeShape7dense_features/occupation_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Gdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Tdense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_table_handleCdense_features/occupation_indicator/to_sparse_input/values:output:0Udense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/occupation_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/occupation_indicator/SparseToDenseSparseToDenseCdense_features/occupation_indicator/to_sparse_input/indices:index:0Hdense_features/occupation_indicator/to_sparse_input/dense_shape:output:0Pdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2:values:0Hdense_features/occupation_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/occupation_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/occupation_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/occupation_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/occupation_indicator/one_hotOneHot9dense_features/occupation_indicator/SparseToDense:dense:0:dense_features/occupation_indicator/one_hot/depth:output:0:dense_features/occupation_indicator/one_hot/Const:output:0<dense_features/occupation_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/occupation_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/occupation_indicator/SumSum4dense_features/occupation_indicator/one_hot:output:0Bdense_features/occupation_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/occupation_indicator/ShapeShape0dense_features/occupation_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/occupation_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/occupation_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/occupation_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/occupation_indicator/strided_sliceStridedSlice2dense_features/occupation_indicator/Shape:output:0@dense_features/occupation_indicator/strided_slice/stack:output:0Bdense_features/occupation_indicator/strided_slice/stack_1:output:0Bdense_features/occupation_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/occupation_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/occupation_indicator/Reshape/shapePack:dense_features/occupation_indicator/strided_slice:output:0<dense_features/occupation_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/occupation_indicator/ReshapeReshape0dense_features/occupation_indicator/Sum:output:0:dense_features/occupation_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????x
-dense_features/topic_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
)dense_features/topic_indicator/ExpandDims
ExpandDimsinputs_topic6dense_features/topic_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????~
=dense_features/topic_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
7dense_features/topic_indicator/to_sparse_input/NotEqualNotEqual2dense_features/topic_indicator/ExpandDims:output:0Fdense_features/topic_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
6dense_features/topic_indicator/to_sparse_input/indicesWhere;dense_features/topic_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
5dense_features/topic_indicator/to_sparse_input/valuesGatherNd2dense_features/topic_indicator/ExpandDims:output:0>dense_features/topic_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
:dense_features/topic_indicator/to_sparse_input/dense_shapeShape2dense_features/topic_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_table_handle>dense_features/topic_indicator/to_sparse_input/values:output:0Pdense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
:dense_features/topic_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
,dense_features/topic_indicator/SparseToDenseSparseToDense>dense_features/topic_indicator/to_sparse_input/indices:index:0Cdense_features/topic_indicator/to_sparse_input/dense_shape:output:0Kdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2:values:0Cdense_features/topic_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????q
,dense_features/topic_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??s
.dense_features/topic_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    n
,dense_features/topic_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
&dense_features/topic_indicator/one_hotOneHot4dense_features/topic_indicator/SparseToDense:dense:05dense_features/topic_indicator/one_hot/depth:output:05dense_features/topic_indicator/one_hot/Const:output:07dense_features/topic_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
4dense_features/topic_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
"dense_features/topic_indicator/SumSum/dense_features/topic_indicator/one_hot:output:0=dense_features/topic_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
$dense_features/topic_indicator/ShapeShape+dense_features/topic_indicator/Sum:output:0*
T0*
_output_shapes
:|
2dense_features/topic_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/topic_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/topic_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,dense_features/topic_indicator/strided_sliceStridedSlice-dense_features/topic_indicator/Shape:output:0;dense_features/topic_indicator/strided_slice/stack:output:0=dense_features/topic_indicator/strided_slice/stack_1:output:0=dense_features/topic_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/topic_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/topic_indicator/Reshape/shapePack5dense_features/topic_indicator/strided_slice:output:07dense_features/topic_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
&dense_features/topic_indicator/ReshapeReshape+dense_features/topic_indicator/Sum:output:05dense_features/topic_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV2.dense_features/age_bucketized/Reshape:output:01dense_features/emotion_indicator/Reshape:output:00dense_features/gender_indicator/Reshape:output:05dense_features/nationality_indicator/Reshape:output:04dense_features/occupation_indicator/Reshape:output:0/dense_features/topic_indicator/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_12/MatMulMatMuldense_features/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOpE^dense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2D^dense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2I^dense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2H^dense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2C^dense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2?
Ddense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2Ddense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV22?
Cdense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2Cdense_features/gender_indicator/hash_table_Lookup/LookupTableFindV22?
Hdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2Hdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV22?
Gdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2Gdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV22?
Bdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2Bdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/age:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/emotion:RN
#
_output_shapes
:?????????
'
_user_specified_nameinputs/gender:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/nationality:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/topic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_4666502
.table_init161_lookuptableimportv2_table_handle*
&table_init161_lookuptableimportv2_keys,
(table_init161_lookuptableimportv2_values	
identity??!table_init161/LookupTableImportV2?
!table_init161/LookupTableImportV2LookupTableImportV2.table_init161_lookuptableimportv2_table_handle&table_init161_lookuptableimportv2_keys(table_init161_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init161/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init161/LookupTableImportV2!table_init161/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?%
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_464984

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
dense_features_464894
dense_features_464896	
dense_features_464898
dense_features_464900	
dense_features_464902
dense_features_464904	
dense_features_464906
dense_features_464908	
dense_features_464910
dense_features_464912	#
dense_12_464927:
??
dense_12_464929:	?#
dense_13_464944:
??
dense_13_464946:	?#
dense_14_464961:
??
dense_14_464963:	?"
dense_15_464978:	?
dense_15_464980:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5dense_features_464894dense_features_464896dense_features_464898dense_features_464900dense_features_464902dense_features_464904dense_features_464906dense_features_464908dense_features_464910dense_features_464912*
Tin
2						*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_464893?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_12_464927dense_12_464929*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_464926?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_464944dense_13_464946*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_464943?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_464961dense_14_464963*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_464960?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_464978dense_15_464980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_464977x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?	
J__inference_dense_features_layer_call_and_return_conditional_losses_465259
features	

features_1

features_2

features_3

features_4

features_5F
Bemotion_indicator_hash_table_lookup_lookuptablefindv2_table_handleG
Cemotion_indicator_hash_table_lookup_lookuptablefindv2_default_value	E
Agender_indicator_hash_table_lookup_lookuptablefindv2_table_handleF
Bgender_indicator_hash_table_lookup_lookuptablefindv2_default_value	J
Fnationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleK
Gnationality_indicator_hash_table_lookup_lookuptablefindv2_default_value	I
Eoccupation_indicator_hash_table_lookup_lookuptablefindv2_table_handleJ
Foccupation_indicator_hash_table_lookup_lookuptablefindv2_default_value	D
@topic_indicator_hash_table_lookup_lookuptablefindv2_table_handleE
Atopic_indicator_hash_table_lookup_lookuptablefindv2_default_value	
identity??5emotion_indicator/hash_table_Lookup/LookupTableFindV2?4gender_indicator/hash_table_Lookup/LookupTableFindV2?9nationality_indicator/hash_table_Lookup/LookupTableFindV2?8occupation_indicator/hash_table_Lookup/LookupTableFindV2?3topic_indicator/hash_table_Lookup/LookupTableFindV2h
age_bucketized/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
age_bucketized/ExpandDims
ExpandDimsfeatures&age_bucketized/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
age_bucketized/CastCast"age_bucketized/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
age_bucketized/Bucketize	Bucketizeage_bucketized/Cast:y:0*
T0*'
_output_shapes
:?????????*^

boundariesP
N"L  ?@   A  pA  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B  ?B  ?B  ?B  ?B  ?B  ?B?
age_bucketized/Cast_1Cast!age_bucketized/Bucketize:output:0*

DstT0	*

SrcT0*'
_output_shapes
:?????????a
age_bucketized/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??c
age_bucketized/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ^
age_bucketized/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
age_bucketized/one_hotOneHotage_bucketized/Cast_1:y:0%age_bucketized/one_hot/depth:output:0%age_bucketized/one_hot/Const:output:0'age_bucketized/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????c
age_bucketized/ShapeShapeage_bucketized/one_hot:output:0*
T0*
_output_shapes
:l
"age_bucketized/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$age_bucketized/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$age_bucketized/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
age_bucketized/strided_sliceStridedSliceage_bucketized/Shape:output:0+age_bucketized/strided_slice/stack:output:0-age_bucketized/strided_slice/stack_1:output:0-age_bucketized/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
age_bucketized/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
age_bucketized/Reshape/shapePack%age_bucketized/strided_slice:output:0'age_bucketized/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
age_bucketized/ReshapeReshapeage_bucketized/one_hot:output:0%age_bucketized/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 emotion_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
emotion_indicator/ExpandDims
ExpandDims
features_1)emotion_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????q
0emotion_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
*emotion_indicator/to_sparse_input/NotEqualNotEqual%emotion_indicator/ExpandDims:output:09emotion_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
)emotion_indicator/to_sparse_input/indicesWhere.emotion_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(emotion_indicator/to_sparse_input/valuesGatherNd%emotion_indicator/ExpandDims:output:01emotion_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
-emotion_indicator/to_sparse_input/dense_shapeShape%emotion_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
5emotion_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Bemotion_indicator_hash_table_lookup_lookuptablefindv2_table_handle1emotion_indicator/to_sparse_input/values:output:0Cemotion_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????x
-emotion_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
emotion_indicator/SparseToDenseSparseToDense1emotion_indicator/to_sparse_input/indices:index:06emotion_indicator/to_sparse_input/dense_shape:output:0>emotion_indicator/hash_table_Lookup/LookupTableFindV2:values:06emotion_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????d
emotion_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??f
!emotion_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    a
emotion_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
emotion_indicator/one_hotOneHot'emotion_indicator/SparseToDense:dense:0(emotion_indicator/one_hot/depth:output:0(emotion_indicator/one_hot/Const:output:0*emotion_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????z
'emotion_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
emotion_indicator/SumSum"emotion_indicator/one_hot:output:00emotion_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????e
emotion_indicator/ShapeShapeemotion_indicator/Sum:output:0*
T0*
_output_shapes
:o
%emotion_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'emotion_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'emotion_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
emotion_indicator/strided_sliceStridedSlice emotion_indicator/Shape:output:0.emotion_indicator/strided_slice/stack:output:00emotion_indicator/strided_slice/stack_1:output:00emotion_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!emotion_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
emotion_indicator/Reshape/shapePack(emotion_indicator/strided_slice:output:0*emotion_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
emotion_indicator/ReshapeReshapeemotion_indicator/Sum:output:0(emotion_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
gender_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gender_indicator/ExpandDims
ExpandDims
features_2(gender_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????p
/gender_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
)gender_indicator/to_sparse_input/NotEqualNotEqual$gender_indicator/ExpandDims:output:08gender_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
(gender_indicator/to_sparse_input/indicesWhere-gender_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
'gender_indicator/to_sparse_input/valuesGatherNd$gender_indicator/ExpandDims:output:00gender_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
,gender_indicator/to_sparse_input/dense_shapeShape$gender_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
4gender_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Agender_indicator_hash_table_lookup_lookuptablefindv2_table_handle0gender_indicator/to_sparse_input/values:output:0Bgender_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????w
,gender_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
gender_indicator/SparseToDenseSparseToDense0gender_indicator/to_sparse_input/indices:index:05gender_indicator/to_sparse_input/dense_shape:output:0=gender_indicator/hash_table_Lookup/LookupTableFindV2:values:05gender_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????c
gender_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
 gender_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    `
gender_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
gender_indicator/one_hotOneHot&gender_indicator/SparseToDense:dense:0'gender_indicator/one_hot/depth:output:0'gender_indicator/one_hot/Const:output:0)gender_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????y
&gender_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
gender_indicator/SumSum!gender_indicator/one_hot:output:0/gender_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????c
gender_indicator/ShapeShapegender_indicator/Sum:output:0*
T0*
_output_shapes
:n
$gender_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&gender_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&gender_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gender_indicator/strided_sliceStridedSlicegender_indicator/Shape:output:0-gender_indicator/strided_slice/stack:output:0/gender_indicator/strided_slice/stack_1:output:0/gender_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 gender_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
gender_indicator/Reshape/shapePack'gender_indicator/strided_slice:output:0)gender_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
gender_indicator/ReshapeReshapegender_indicator/Sum:output:0'gender_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$nationality_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 nationality_indicator/ExpandDims
ExpandDims
features_3-nationality_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4nationality_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.nationality_indicator/to_sparse_input/NotEqualNotEqual)nationality_indicator/ExpandDims:output:0=nationality_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-nationality_indicator/to_sparse_input/indicesWhere2nationality_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,nationality_indicator/to_sparse_input/valuesGatherNd)nationality_indicator/ExpandDims:output:05nationality_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1nationality_indicator/to_sparse_input/dense_shapeShape)nationality_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
9nationality_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Fnationality_indicator_hash_table_lookup_lookuptablefindv2_table_handle5nationality_indicator/to_sparse_input/values:output:0Gnationality_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1nationality_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#nationality_indicator/SparseToDenseSparseToDense5nationality_indicator/to_sparse_input/indices:index:0:nationality_indicator/to_sparse_input/dense_shape:output:0Bnationality_indicator/hash_table_Lookup/LookupTableFindV2:values:0:nationality_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#nationality_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%nationality_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#nationality_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
nationality_indicator/one_hotOneHot+nationality_indicator/SparseToDense:dense:0,nationality_indicator/one_hot/depth:output:0,nationality_indicator/one_hot/Const:output:0.nationality_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+nationality_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
nationality_indicator/SumSum&nationality_indicator/one_hot:output:04nationality_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
nationality_indicator/ShapeShape"nationality_indicator/Sum:output:0*
T0*
_output_shapes
:s
)nationality_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+nationality_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+nationality_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#nationality_indicator/strided_sliceStridedSlice$nationality_indicator/Shape:output:02nationality_indicator/strided_slice/stack:output:04nationality_indicator/strided_slice/stack_1:output:04nationality_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%nationality_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#nationality_indicator/Reshape/shapePack,nationality_indicator/strided_slice:output:0.nationality_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
nationality_indicator/ReshapeReshape"nationality_indicator/Sum:output:0,nationality_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????n
#occupation_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
occupation_indicator/ExpandDims
ExpandDims
features_4,occupation_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3occupation_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-occupation_indicator/to_sparse_input/NotEqualNotEqual(occupation_indicator/ExpandDims:output:0<occupation_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,occupation_indicator/to_sparse_input/indicesWhere1occupation_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+occupation_indicator/to_sparse_input/valuesGatherNd(occupation_indicator/ExpandDims:output:04occupation_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0occupation_indicator/to_sparse_input/dense_shapeShape(occupation_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
8occupation_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Eoccupation_indicator_hash_table_lookup_lookuptablefindv2_table_handle4occupation_indicator/to_sparse_input/values:output:0Foccupation_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0occupation_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"occupation_indicator/SparseToDenseSparseToDense4occupation_indicator/to_sparse_input/indices:index:09occupation_indicator/to_sparse_input/dense_shape:output:0Aoccupation_indicator/hash_table_Lookup/LookupTableFindV2:values:09occupation_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"occupation_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$occupation_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"occupation_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
occupation_indicator/one_hotOneHot*occupation_indicator/SparseToDense:dense:0+occupation_indicator/one_hot/depth:output:0+occupation_indicator/one_hot/Const:output:0-occupation_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*occupation_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
occupation_indicator/SumSum%occupation_indicator/one_hot:output:03occupation_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
occupation_indicator/ShapeShape!occupation_indicator/Sum:output:0*
T0*
_output_shapes
:r
(occupation_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*occupation_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*occupation_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"occupation_indicator/strided_sliceStridedSlice#occupation_indicator/Shape:output:01occupation_indicator/strided_slice/stack:output:03occupation_indicator/strided_slice/stack_1:output:03occupation_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$occupation_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"occupation_indicator/Reshape/shapePack+occupation_indicator/strided_slice:output:0-occupation_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
occupation_indicator/ReshapeReshape!occupation_indicator/Sum:output:0+occupation_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
topic_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
topic_indicator/ExpandDims
ExpandDims
features_5'topic_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????o
.topic_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
(topic_indicator/to_sparse_input/NotEqualNotEqual#topic_indicator/ExpandDims:output:07topic_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
'topic_indicator/to_sparse_input/indicesWhere,topic_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
&topic_indicator/to_sparse_input/valuesGatherNd#topic_indicator/ExpandDims:output:0/topic_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
+topic_indicator/to_sparse_input/dense_shapeShape#topic_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3topic_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@topic_indicator_hash_table_lookup_lookuptablefindv2_table_handle/topic_indicator/to_sparse_input/values:output:0Atopic_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????v
+topic_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
topic_indicator/SparseToDenseSparseToDense/topic_indicator/to_sparse_input/indices:index:04topic_indicator/to_sparse_input/dense_shape:output:0<topic_indicator/hash_table_Lookup/LookupTableFindV2:values:04topic_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????b
topic_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
topic_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    _
topic_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
topic_indicator/one_hotOneHot%topic_indicator/SparseToDense:dense:0&topic_indicator/one_hot/depth:output:0&topic_indicator/one_hot/Const:output:0(topic_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????x
%topic_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
topic_indicator/SumSum topic_indicator/one_hot:output:0.topic_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????a
topic_indicator/ShapeShapetopic_indicator/Sum:output:0*
T0*
_output_shapes
:m
#topic_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%topic_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%topic_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
topic_indicator/strided_sliceStridedSlicetopic_indicator/Shape:output:0,topic_indicator/strided_slice/stack:output:0.topic_indicator/strided_slice/stack_1:output:0.topic_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
topic_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
topic_indicator/Reshape/shapePack&topic_indicator/strided_slice:output:0(topic_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
topic_indicator/ReshapeReshapetopic_indicator/Sum:output:0&topic_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2age_bucketized/Reshape:output:0"emotion_indicator/Reshape:output:0!gender_indicator/Reshape:output:0&nationality_indicator/Reshape:output:0%occupation_indicator/Reshape:output:0 topic_indicator/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp6^emotion_indicator/hash_table_Lookup/LookupTableFindV25^gender_indicator/hash_table_Lookup/LookupTableFindV2:^nationality_indicator/hash_table_Lookup/LookupTableFindV29^occupation_indicator/hash_table_Lookup/LookupTableFindV24^topic_indicator/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2n
5emotion_indicator/hash_table_Lookup/LookupTableFindV25emotion_indicator/hash_table_Lookup/LookupTableFindV22l
4gender_indicator/hash_table_Lookup/LookupTableFindV24gender_indicator/hash_table_Lookup/LookupTableFindV22v
9nationality_indicator/hash_table_Lookup/LookupTableFindV29nationality_indicator/hash_table_Lookup/LookupTableFindV22t
8occupation_indicator/hash_table_Lookup/LookupTableFindV28occupation_indicator/hash_table_Lookup/LookupTableFindV22j
3topic_indicator/hash_table_Lookup/LookupTableFindV23topic_indicator/hash_table_Lookup/LookupTableFindV2:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_466088

inputs_age	
inputs_emotion
inputs_gender
inputs_nationality
inputs_occupation
inputs_topicU
Qdense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_table_handleV
Rdense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_default_value	T
Pdense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_table_handleU
Qdense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_default_value	Y
Udense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleZ
Vdense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_default_value	X
Tdense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_table_handleY
Udense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_default_value	S
Odense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_table_handleT
Pdense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_default_value	;
'dense_12_matmul_readvariableop_resource:
??7
(dense_12_biasadd_readvariableop_resource:	?;
'dense_13_matmul_readvariableop_resource:
??7
(dense_13_biasadd_readvariableop_resource:	?;
'dense_14_matmul_readvariableop_resource:
??7
(dense_14_biasadd_readvariableop_resource:	?:
'dense_15_matmul_readvariableop_resource:	?6
(dense_15_biasadd_readvariableop_resource:
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?Ddense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2?Cdense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2?Hdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2?Gdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2?Bdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2w
,dense_features/age_bucketized/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(dense_features/age_bucketized/ExpandDims
ExpandDims
inputs_age5dense_features/age_bucketized/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
"dense_features/age_bucketized/CastCast1dense_features/age_bucketized/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
'dense_features/age_bucketized/Bucketize	Bucketize&dense_features/age_bucketized/Cast:y:0*
T0*'
_output_shapes
:?????????*^

boundariesP
N"L  ?@   A  pA  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B  ?B  ?B  ?B  ?B  ?B  ?B?
$dense_features/age_bucketized/Cast_1Cast0dense_features/age_bucketized/Bucketize:output:0*

DstT0	*

SrcT0*'
_output_shapes
:?????????p
+dense_features/age_bucketized/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??r
-dense_features/age_bucketized/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    m
+dense_features/age_bucketized/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
%dense_features/age_bucketized/one_hotOneHot(dense_features/age_bucketized/Cast_1:y:04dense_features/age_bucketized/one_hot/depth:output:04dense_features/age_bucketized/one_hot/Const:output:06dense_features/age_bucketized/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
#dense_features/age_bucketized/ShapeShape.dense_features/age_bucketized/one_hot:output:0*
T0*
_output_shapes
:{
1dense_features/age_bucketized/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_features/age_bucketized/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_features/age_bucketized/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_features/age_bucketized/strided_sliceStridedSlice,dense_features/age_bucketized/Shape:output:0:dense_features/age_bucketized/strided_slice/stack:output:0<dense_features/age_bucketized/strided_slice/stack_1:output:0<dense_features/age_bucketized/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-dense_features/age_bucketized/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/age_bucketized/Reshape/shapePack4dense_features/age_bucketized/strided_slice:output:06dense_features/age_bucketized/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
%dense_features/age_bucketized/ReshapeReshape.dense_features/age_bucketized/one_hot:output:04dense_features/age_bucketized/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/dense_features/emotion_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+dense_features/emotion_indicator/ExpandDims
ExpandDimsinputs_emotion8dense_features/emotion_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
?dense_features/emotion_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
9dense_features/emotion_indicator/to_sparse_input/NotEqualNotEqual4dense_features/emotion_indicator/ExpandDims:output:0Hdense_features/emotion_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
8dense_features/emotion_indicator/to_sparse_input/indicesWhere=dense_features/emotion_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
7dense_features/emotion_indicator/to_sparse_input/valuesGatherNd4dense_features/emotion_indicator/ExpandDims:output:0@dense_features/emotion_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
<dense_features/emotion_indicator/to_sparse_input/dense_shapeShape4dense_features/emotion_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Ddense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qdense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_table_handle@dense_features/emotion_indicator/to_sparse_input/values:output:0Rdense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
<dense_features/emotion_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
.dense_features/emotion_indicator/SparseToDenseSparseToDense@dense_features/emotion_indicator/to_sparse_input/indices:index:0Edense_features/emotion_indicator/to_sparse_input/dense_shape:output:0Mdense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2:values:0Edense_features/emotion_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????s
.dense_features/emotion_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
0dense_features/emotion_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    p
.dense_features/emotion_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
(dense_features/emotion_indicator/one_hotOneHot6dense_features/emotion_indicator/SparseToDense:dense:07dense_features/emotion_indicator/one_hot/depth:output:07dense_features/emotion_indicator/one_hot/Const:output:09dense_features/emotion_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
6dense_features/emotion_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
$dense_features/emotion_indicator/SumSum1dense_features/emotion_indicator/one_hot:output:0?dense_features/emotion_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
&dense_features/emotion_indicator/ShapeShape-dense_features/emotion_indicator/Sum:output:0*
T0*
_output_shapes
:~
4dense_features/emotion_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/emotion_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/emotion_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/emotion_indicator/strided_sliceStridedSlice/dense_features/emotion_indicator/Shape:output:0=dense_features/emotion_indicator/strided_slice/stack:output:0?dense_features/emotion_indicator/strided_slice/stack_1:output:0?dense_features/emotion_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/emotion_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/emotion_indicator/Reshape/shapePack7dense_features/emotion_indicator/strided_slice:output:09dense_features/emotion_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/emotion_indicator/ReshapeReshape-dense_features/emotion_indicator/Sum:output:07dense_features/emotion_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
.dense_features/gender_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
*dense_features/gender_indicator/ExpandDims
ExpandDimsinputs_gender7dense_features/gender_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????
>dense_features/gender_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
8dense_features/gender_indicator/to_sparse_input/NotEqualNotEqual3dense_features/gender_indicator/ExpandDims:output:0Gdense_features/gender_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
7dense_features/gender_indicator/to_sparse_input/indicesWhere<dense_features/gender_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
6dense_features/gender_indicator/to_sparse_input/valuesGatherNd3dense_features/gender_indicator/ExpandDims:output:0?dense_features/gender_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
;dense_features/gender_indicator/to_sparse_input/dense_shapeShape3dense_features/gender_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Cdense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Pdense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_table_handle?dense_features/gender_indicator/to_sparse_input/values:output:0Qdense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
;dense_features/gender_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
-dense_features/gender_indicator/SparseToDenseSparseToDense?dense_features/gender_indicator/to_sparse_input/indices:index:0Ddense_features/gender_indicator/to_sparse_input/dense_shape:output:0Ldense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2:values:0Ddense_features/gender_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????r
-dense_features/gender_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
/dense_features/gender_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    o
-dense_features/gender_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
'dense_features/gender_indicator/one_hotOneHot5dense_features/gender_indicator/SparseToDense:dense:06dense_features/gender_indicator/one_hot/depth:output:06dense_features/gender_indicator/one_hot/Const:output:08dense_features/gender_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
5dense_features/gender_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
#dense_features/gender_indicator/SumSum0dense_features/gender_indicator/one_hot:output:0>dense_features/gender_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
%dense_features/gender_indicator/ShapeShape,dense_features/gender_indicator/Sum:output:0*
T0*
_output_shapes
:}
3dense_features/gender_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/gender_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/gender_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_features/gender_indicator/strided_sliceStridedSlice.dense_features/gender_indicator/Shape:output:0<dense_features/gender_indicator/strided_slice/stack:output:0>dense_features/gender_indicator/strided_slice/stack_1:output:0>dense_features/gender_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/gender_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
-dense_features/gender_indicator/Reshape/shapePack6dense_features/gender_indicator/strided_slice:output:08dense_features/gender_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
'dense_features/gender_indicator/ReshapeReshape,dense_features/gender_indicator/Sum:output:06dense_features/gender_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3dense_features/nationality_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/dense_features/nationality_indicator/ExpandDims
ExpandDimsinputs_nationality<dense_features/nationality_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/nationality_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/nationality_indicator/to_sparse_input/NotEqualNotEqual8dense_features/nationality_indicator/ExpandDims:output:0Ldense_features/nationality_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/nationality_indicator/to_sparse_input/indicesWhereAdense_features/nationality_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/nationality_indicator/to_sparse_input/valuesGatherNd8dense_features/nationality_indicator/ExpandDims:output:0Ddense_features/nationality_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/nationality_indicator/to_sparse_input/dense_shapeShape8dense_features/nationality_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Hdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Udense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleDdense_features/nationality_indicator/to_sparse_input/values:output:0Vdense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/nationality_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/nationality_indicator/SparseToDenseSparseToDenseDdense_features/nationality_indicator/to_sparse_input/indices:index:0Idense_features/nationality_indicator/to_sparse_input/dense_shape:output:0Qdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2:values:0Idense_features/nationality_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/nationality_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/nationality_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    u
2dense_features/nationality_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
,dense_features/nationality_indicator/one_hotOneHot:dense_features/nationality_indicator/SparseToDense:dense:0;dense_features/nationality_indicator/one_hot/depth:output:0;dense_features/nationality_indicator/one_hot/Const:output:0=dense_features/nationality_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
:dense_features/nationality_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/nationality_indicator/SumSum5dense_features/nationality_indicator/one_hot:output:0Cdense_features/nationality_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
*dense_features/nationality_indicator/ShapeShape1dense_features/nationality_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/nationality_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/nationality_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/nationality_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/nationality_indicator/strided_sliceStridedSlice3dense_features/nationality_indicator/Shape:output:0Adense_features/nationality_indicator/strided_slice/stack:output:0Cdense_features/nationality_indicator/strided_slice/stack_1:output:0Cdense_features/nationality_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
4dense_features/nationality_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
2dense_features/nationality_indicator/Reshape/shapePack;dense_features/nationality_indicator/strided_slice:output:0=dense_features/nationality_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/nationality_indicator/ReshapeReshape1dense_features/nationality_indicator/Sum:output:0;dense_features/nationality_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????}
2dense_features/occupation_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/occupation_indicator/ExpandDims
ExpandDimsinputs_occupation;dense_features/occupation_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/occupation_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/occupation_indicator/to_sparse_input/NotEqualNotEqual7dense_features/occupation_indicator/ExpandDims:output:0Kdense_features/occupation_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/occupation_indicator/to_sparse_input/indicesWhere@dense_features/occupation_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/occupation_indicator/to_sparse_input/valuesGatherNd7dense_features/occupation_indicator/ExpandDims:output:0Cdense_features/occupation_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/occupation_indicator/to_sparse_input/dense_shapeShape7dense_features/occupation_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Gdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Tdense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_table_handleCdense_features/occupation_indicator/to_sparse_input/values:output:0Udense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/occupation_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/occupation_indicator/SparseToDenseSparseToDenseCdense_features/occupation_indicator/to_sparse_input/indices:index:0Hdense_features/occupation_indicator/to_sparse_input/dense_shape:output:0Pdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2:values:0Hdense_features/occupation_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/occupation_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/occupation_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/occupation_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/occupation_indicator/one_hotOneHot9dense_features/occupation_indicator/SparseToDense:dense:0:dense_features/occupation_indicator/one_hot/depth:output:0:dense_features/occupation_indicator/one_hot/Const:output:0<dense_features/occupation_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/occupation_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/occupation_indicator/SumSum4dense_features/occupation_indicator/one_hot:output:0Bdense_features/occupation_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/occupation_indicator/ShapeShape0dense_features/occupation_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/occupation_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/occupation_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/occupation_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/occupation_indicator/strided_sliceStridedSlice2dense_features/occupation_indicator/Shape:output:0@dense_features/occupation_indicator/strided_slice/stack:output:0Bdense_features/occupation_indicator/strided_slice/stack_1:output:0Bdense_features/occupation_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/occupation_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/occupation_indicator/Reshape/shapePack:dense_features/occupation_indicator/strided_slice:output:0<dense_features/occupation_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/occupation_indicator/ReshapeReshape0dense_features/occupation_indicator/Sum:output:0:dense_features/occupation_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????x
-dense_features/topic_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
)dense_features/topic_indicator/ExpandDims
ExpandDimsinputs_topic6dense_features/topic_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????~
=dense_features/topic_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
7dense_features/topic_indicator/to_sparse_input/NotEqualNotEqual2dense_features/topic_indicator/ExpandDims:output:0Fdense_features/topic_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
6dense_features/topic_indicator/to_sparse_input/indicesWhere;dense_features/topic_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
5dense_features/topic_indicator/to_sparse_input/valuesGatherNd2dense_features/topic_indicator/ExpandDims:output:0>dense_features/topic_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
:dense_features/topic_indicator/to_sparse_input/dense_shapeShape2dense_features/topic_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_table_handle>dense_features/topic_indicator/to_sparse_input/values:output:0Pdense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
:dense_features/topic_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
,dense_features/topic_indicator/SparseToDenseSparseToDense>dense_features/topic_indicator/to_sparse_input/indices:index:0Cdense_features/topic_indicator/to_sparse_input/dense_shape:output:0Kdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2:values:0Cdense_features/topic_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????q
,dense_features/topic_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??s
.dense_features/topic_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    n
,dense_features/topic_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
&dense_features/topic_indicator/one_hotOneHot4dense_features/topic_indicator/SparseToDense:dense:05dense_features/topic_indicator/one_hot/depth:output:05dense_features/topic_indicator/one_hot/Const:output:07dense_features/topic_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
4dense_features/topic_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
"dense_features/topic_indicator/SumSum/dense_features/topic_indicator/one_hot:output:0=dense_features/topic_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
$dense_features/topic_indicator/ShapeShape+dense_features/topic_indicator/Sum:output:0*
T0*
_output_shapes
:|
2dense_features/topic_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/topic_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/topic_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,dense_features/topic_indicator/strided_sliceStridedSlice-dense_features/topic_indicator/Shape:output:0;dense_features/topic_indicator/strided_slice/stack:output:0=dense_features/topic_indicator/strided_slice/stack_1:output:0=dense_features/topic_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/topic_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/topic_indicator/Reshape/shapePack5dense_features/topic_indicator/strided_slice:output:07dense_features/topic_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
&dense_features/topic_indicator/ReshapeReshape+dense_features/topic_indicator/Sum:output:05dense_features/topic_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV2.dense_features/age_bucketized/Reshape:output:01dense_features/emotion_indicator/Reshape:output:00dense_features/gender_indicator/Reshape:output:05dense_features/nationality_indicator/Reshape:output:04dense_features/occupation_indicator/Reshape:output:0/dense_features/topic_indicator/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_12/MatMulMatMuldense_features/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOpE^dense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2D^dense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2I^dense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2H^dense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2C^dense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2?
Ddense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2Ddense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV22?
Cdense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2Cdense_features/gender_indicator/hash_table_Lookup/LookupTableFindV22?
Hdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2Hdense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV22?
Gdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2Gdense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV22?
Bdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2Bdense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/age:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/emotion:RN
#
_output_shapes
:?????????
'
_user_specified_nameinputs/gender:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/nationality:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/topic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_dense_13_layer_call_fn_466493

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_464943p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_12_layer_call_and_return_conditional_losses_466484

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_4666582
.table_init197_lookuptableimportv2_table_handle*
&table_init197_lookuptableimportv2_keys,
(table_init197_lookuptableimportv2_values	
identity??!table_init197/LookupTableImportV2?
!table_init197/LookupTableImportV2LookupTableImportV2.table_init197_lookuptableimportv2_table_handle&table_init197_lookuptableimportv2_keys(table_init197_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init197/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2F
!table_init197/LookupTableImportV2!table_init197/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
-
__inference__destroyer_466580
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
/__inference_dense_features_layer_call_fn_466118
features_age	
features_emotion
features_gender
features_nationality
features_occupation
features_topic
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_agefeatures_emotionfeatures_genderfeatures_nationalityfeatures_occupationfeatures_topicunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2						*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_464893p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
#
_output_shapes
:?????????
&
_user_specified_namefeatures/age:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/emotion:TP
#
_output_shapes
:?????????
)
_user_specified_namefeatures/gender:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/nationality:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/occupation:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/topic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_4666742
.table_init269_lookuptableimportv2_table_handle*
&table_init269_lookuptableimportv2_keys,
(table_init269_lookuptableimportv2_values	
identity??!table_init269/LookupTableImportV2?
!table_init269/LookupTableImportV2LookupTableImportV2.table_init269_lookuptableimportv2_table_handle&table_init269_lookuptableimportv2_keys(table_init269_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init269/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init269/LookupTableImportV2!table_init269/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?

?
D__inference_dense_15_layer_call_and_return_conditional_losses_464977

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_465624
age	
emotion

gender
nationality

occupation	
topic
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9:
??

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallageemotiongendernationality
occupationtopicunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*#
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_464718o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:?????????

_user_specified_nameage:LH
#
_output_shapes
:?????????
!
_user_specified_name	emotion:KG
#
_output_shapes
:?????????
 
_user_specified_namegender:PL
#
_output_shapes
:?????????
%
_user_specified_namenationality:OK
#
_output_shapes
:?????????
$
_user_specified_name
occupation:JF
#
_output_shapes
:?????????

_user_specified_nametopic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
"__inference__traced_restore_466935
file_prefixA
-assignvariableop_sequential_3_dense_12_kernel:
??<
-assignvariableop_1_sequential_3_dense_12_bias:	?C
/assignvariableop_2_sequential_3_dense_13_kernel:
??<
-assignvariableop_3_sequential_3_dense_13_bias:	?C
/assignvariableop_4_sequential_3_dense_14_kernel:
??<
-assignvariableop_5_sequential_3_dense_14_bias:	?B
/assignvariableop_6_sequential_3_dense_15_kernel:	?;
-assignvariableop_7_sequential_3_dense_15_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: K
7assignvariableop_17_adam_sequential_3_dense_12_kernel_m:
??D
5assignvariableop_18_adam_sequential_3_dense_12_bias_m:	?K
7assignvariableop_19_adam_sequential_3_dense_13_kernel_m:
??D
5assignvariableop_20_adam_sequential_3_dense_13_bias_m:	?K
7assignvariableop_21_adam_sequential_3_dense_14_kernel_m:
??D
5assignvariableop_22_adam_sequential_3_dense_14_bias_m:	?J
7assignvariableop_23_adam_sequential_3_dense_15_kernel_m:	?C
5assignvariableop_24_adam_sequential_3_dense_15_bias_m:K
7assignvariableop_25_adam_sequential_3_dense_12_kernel_v:
??D
5assignvariableop_26_adam_sequential_3_dense_12_bias_v:	?K
7assignvariableop_27_adam_sequential_3_dense_13_kernel_v:
??D
5assignvariableop_28_adam_sequential_3_dense_13_bias_v:	?K
7assignvariableop_29_adam_sequential_3_dense_14_kernel_v:
??D
5assignvariableop_30_adam_sequential_3_dense_14_bias_v:	?J
7assignvariableop_31_adam_sequential_3_dense_15_kernel_v:	?C
5assignvariableop_32_adam_sequential_3_dense_15_bias_v:
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp-assignvariableop_sequential_3_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_sequential_3_dense_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_sequential_3_dense_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_sequential_3_dense_13_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_sequential_3_dense_14_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp-assignvariableop_5_sequential_3_dense_14_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_sequential_3_dense_15_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_sequential_3_dense_15_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adam_sequential_3_dense_12_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adam_sequential_3_dense_12_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_sequential_3_dense_13_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_sequential_3_dense_13_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_sequential_3_dense_14_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_sequential_3_dense_14_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adam_sequential_3_dense_15_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_sequential_3_dense_15_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_sequential_3_dense_12_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_sequential_3_dense_12_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_sequential_3_dense_13_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_sequential_3_dense_13_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_sequential_3_dense_14_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_sequential_3_dense_14_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_sequential_3_dense_15_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_sequential_3_dense_15_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
?%
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_465385

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
dense_features_465343
dense_features_465345	
dense_features_465347
dense_features_465349	
dense_features_465351
dense_features_465353	
dense_features_465355
dense_features_465357	
dense_features_465359
dense_features_465361	#
dense_12_465364:
??
dense_12_465366:	?#
dense_13_465369:
??
dense_13_465371:	?#
dense_14_465374:
??
dense_14_465376:	?"
dense_15_465379:	?
dense_15_465381:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5dense_features_465343dense_features_465345dense_features_465347dense_features_465349dense_features_465351dense_features_465353dense_features_465355dense_features_465357dense_features_465359dense_features_465361*
Tin
2						*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_465259?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_12_465364dense_12_465366*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_464926?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_465369dense_13_465371*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_464943?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_465374dense_14_465376*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_464960?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_465379dense_15_465381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_464977x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?	
J__inference_dense_features_layer_call_and_return_conditional_losses_466464
features_age	
features_emotion
features_gender
features_nationality
features_occupation
features_topicF
Bemotion_indicator_hash_table_lookup_lookuptablefindv2_table_handleG
Cemotion_indicator_hash_table_lookup_lookuptablefindv2_default_value	E
Agender_indicator_hash_table_lookup_lookuptablefindv2_table_handleF
Bgender_indicator_hash_table_lookup_lookuptablefindv2_default_value	J
Fnationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleK
Gnationality_indicator_hash_table_lookup_lookuptablefindv2_default_value	I
Eoccupation_indicator_hash_table_lookup_lookuptablefindv2_table_handleJ
Foccupation_indicator_hash_table_lookup_lookuptablefindv2_default_value	D
@topic_indicator_hash_table_lookup_lookuptablefindv2_table_handleE
Atopic_indicator_hash_table_lookup_lookuptablefindv2_default_value	
identity??5emotion_indicator/hash_table_Lookup/LookupTableFindV2?4gender_indicator/hash_table_Lookup/LookupTableFindV2?9nationality_indicator/hash_table_Lookup/LookupTableFindV2?8occupation_indicator/hash_table_Lookup/LookupTableFindV2?3topic_indicator/hash_table_Lookup/LookupTableFindV2h
age_bucketized/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
age_bucketized/ExpandDims
ExpandDimsfeatures_age&age_bucketized/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
age_bucketized/CastCast"age_bucketized/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
age_bucketized/Bucketize	Bucketizeage_bucketized/Cast:y:0*
T0*'
_output_shapes
:?????????*^

boundariesP
N"L  ?@   A  pA  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B  ?B  ?B  ?B  ?B  ?B  ?B?
age_bucketized/Cast_1Cast!age_bucketized/Bucketize:output:0*

DstT0	*

SrcT0*'
_output_shapes
:?????????a
age_bucketized/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??c
age_bucketized/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ^
age_bucketized/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
age_bucketized/one_hotOneHotage_bucketized/Cast_1:y:0%age_bucketized/one_hot/depth:output:0%age_bucketized/one_hot/Const:output:0'age_bucketized/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????c
age_bucketized/ShapeShapeage_bucketized/one_hot:output:0*
T0*
_output_shapes
:l
"age_bucketized/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$age_bucketized/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$age_bucketized/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
age_bucketized/strided_sliceStridedSliceage_bucketized/Shape:output:0+age_bucketized/strided_slice/stack:output:0-age_bucketized/strided_slice/stack_1:output:0-age_bucketized/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
age_bucketized/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
age_bucketized/Reshape/shapePack%age_bucketized/strided_slice:output:0'age_bucketized/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
age_bucketized/ReshapeReshapeage_bucketized/one_hot:output:0%age_bucketized/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 emotion_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
emotion_indicator/ExpandDims
ExpandDimsfeatures_emotion)emotion_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????q
0emotion_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
*emotion_indicator/to_sparse_input/NotEqualNotEqual%emotion_indicator/ExpandDims:output:09emotion_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
)emotion_indicator/to_sparse_input/indicesWhere.emotion_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(emotion_indicator/to_sparse_input/valuesGatherNd%emotion_indicator/ExpandDims:output:01emotion_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
-emotion_indicator/to_sparse_input/dense_shapeShape%emotion_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
5emotion_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Bemotion_indicator_hash_table_lookup_lookuptablefindv2_table_handle1emotion_indicator/to_sparse_input/values:output:0Cemotion_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????x
-emotion_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
emotion_indicator/SparseToDenseSparseToDense1emotion_indicator/to_sparse_input/indices:index:06emotion_indicator/to_sparse_input/dense_shape:output:0>emotion_indicator/hash_table_Lookup/LookupTableFindV2:values:06emotion_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????d
emotion_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??f
!emotion_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    a
emotion_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
emotion_indicator/one_hotOneHot'emotion_indicator/SparseToDense:dense:0(emotion_indicator/one_hot/depth:output:0(emotion_indicator/one_hot/Const:output:0*emotion_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????z
'emotion_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
emotion_indicator/SumSum"emotion_indicator/one_hot:output:00emotion_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????e
emotion_indicator/ShapeShapeemotion_indicator/Sum:output:0*
T0*
_output_shapes
:o
%emotion_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'emotion_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'emotion_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
emotion_indicator/strided_sliceStridedSlice emotion_indicator/Shape:output:0.emotion_indicator/strided_slice/stack:output:00emotion_indicator/strided_slice/stack_1:output:00emotion_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!emotion_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
emotion_indicator/Reshape/shapePack(emotion_indicator/strided_slice:output:0*emotion_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
emotion_indicator/ReshapeReshapeemotion_indicator/Sum:output:0(emotion_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
gender_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gender_indicator/ExpandDims
ExpandDimsfeatures_gender(gender_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????p
/gender_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
)gender_indicator/to_sparse_input/NotEqualNotEqual$gender_indicator/ExpandDims:output:08gender_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
(gender_indicator/to_sparse_input/indicesWhere-gender_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
'gender_indicator/to_sparse_input/valuesGatherNd$gender_indicator/ExpandDims:output:00gender_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
,gender_indicator/to_sparse_input/dense_shapeShape$gender_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
4gender_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Agender_indicator_hash_table_lookup_lookuptablefindv2_table_handle0gender_indicator/to_sparse_input/values:output:0Bgender_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????w
,gender_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
gender_indicator/SparseToDenseSparseToDense0gender_indicator/to_sparse_input/indices:index:05gender_indicator/to_sparse_input/dense_shape:output:0=gender_indicator/hash_table_Lookup/LookupTableFindV2:values:05gender_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????c
gender_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
 gender_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    `
gender_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
gender_indicator/one_hotOneHot&gender_indicator/SparseToDense:dense:0'gender_indicator/one_hot/depth:output:0'gender_indicator/one_hot/Const:output:0)gender_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????y
&gender_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
gender_indicator/SumSum!gender_indicator/one_hot:output:0/gender_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????c
gender_indicator/ShapeShapegender_indicator/Sum:output:0*
T0*
_output_shapes
:n
$gender_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&gender_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&gender_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gender_indicator/strided_sliceStridedSlicegender_indicator/Shape:output:0-gender_indicator/strided_slice/stack:output:0/gender_indicator/strided_slice/stack_1:output:0/gender_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 gender_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
gender_indicator/Reshape/shapePack'gender_indicator/strided_slice:output:0)gender_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
gender_indicator/ReshapeReshapegender_indicator/Sum:output:0'gender_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$nationality_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 nationality_indicator/ExpandDims
ExpandDimsfeatures_nationality-nationality_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4nationality_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.nationality_indicator/to_sparse_input/NotEqualNotEqual)nationality_indicator/ExpandDims:output:0=nationality_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-nationality_indicator/to_sparse_input/indicesWhere2nationality_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,nationality_indicator/to_sparse_input/valuesGatherNd)nationality_indicator/ExpandDims:output:05nationality_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1nationality_indicator/to_sparse_input/dense_shapeShape)nationality_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
9nationality_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Fnationality_indicator_hash_table_lookup_lookuptablefindv2_table_handle5nationality_indicator/to_sparse_input/values:output:0Gnationality_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1nationality_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#nationality_indicator/SparseToDenseSparseToDense5nationality_indicator/to_sparse_input/indices:index:0:nationality_indicator/to_sparse_input/dense_shape:output:0Bnationality_indicator/hash_table_Lookup/LookupTableFindV2:values:0:nationality_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#nationality_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%nationality_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#nationality_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
nationality_indicator/one_hotOneHot+nationality_indicator/SparseToDense:dense:0,nationality_indicator/one_hot/depth:output:0,nationality_indicator/one_hot/Const:output:0.nationality_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+nationality_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
nationality_indicator/SumSum&nationality_indicator/one_hot:output:04nationality_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
nationality_indicator/ShapeShape"nationality_indicator/Sum:output:0*
T0*
_output_shapes
:s
)nationality_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+nationality_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+nationality_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#nationality_indicator/strided_sliceStridedSlice$nationality_indicator/Shape:output:02nationality_indicator/strided_slice/stack:output:04nationality_indicator/strided_slice/stack_1:output:04nationality_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%nationality_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#nationality_indicator/Reshape/shapePack,nationality_indicator/strided_slice:output:0.nationality_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
nationality_indicator/ReshapeReshape"nationality_indicator/Sum:output:0,nationality_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????n
#occupation_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
occupation_indicator/ExpandDims
ExpandDimsfeatures_occupation,occupation_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3occupation_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-occupation_indicator/to_sparse_input/NotEqualNotEqual(occupation_indicator/ExpandDims:output:0<occupation_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,occupation_indicator/to_sparse_input/indicesWhere1occupation_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+occupation_indicator/to_sparse_input/valuesGatherNd(occupation_indicator/ExpandDims:output:04occupation_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0occupation_indicator/to_sparse_input/dense_shapeShape(occupation_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
8occupation_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Eoccupation_indicator_hash_table_lookup_lookuptablefindv2_table_handle4occupation_indicator/to_sparse_input/values:output:0Foccupation_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0occupation_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"occupation_indicator/SparseToDenseSparseToDense4occupation_indicator/to_sparse_input/indices:index:09occupation_indicator/to_sparse_input/dense_shape:output:0Aoccupation_indicator/hash_table_Lookup/LookupTableFindV2:values:09occupation_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"occupation_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$occupation_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"occupation_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
occupation_indicator/one_hotOneHot*occupation_indicator/SparseToDense:dense:0+occupation_indicator/one_hot/depth:output:0+occupation_indicator/one_hot/Const:output:0-occupation_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*occupation_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
occupation_indicator/SumSum%occupation_indicator/one_hot:output:03occupation_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
occupation_indicator/ShapeShape!occupation_indicator/Sum:output:0*
T0*
_output_shapes
:r
(occupation_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*occupation_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*occupation_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"occupation_indicator/strided_sliceStridedSlice#occupation_indicator/Shape:output:01occupation_indicator/strided_slice/stack:output:03occupation_indicator/strided_slice/stack_1:output:03occupation_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$occupation_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"occupation_indicator/Reshape/shapePack+occupation_indicator/strided_slice:output:0-occupation_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
occupation_indicator/ReshapeReshape!occupation_indicator/Sum:output:0+occupation_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
topic_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
topic_indicator/ExpandDims
ExpandDimsfeatures_topic'topic_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????o
.topic_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
(topic_indicator/to_sparse_input/NotEqualNotEqual#topic_indicator/ExpandDims:output:07topic_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
'topic_indicator/to_sparse_input/indicesWhere,topic_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
&topic_indicator/to_sparse_input/valuesGatherNd#topic_indicator/ExpandDims:output:0/topic_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
+topic_indicator/to_sparse_input/dense_shapeShape#topic_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3topic_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@topic_indicator_hash_table_lookup_lookuptablefindv2_table_handle/topic_indicator/to_sparse_input/values:output:0Atopic_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????v
+topic_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
topic_indicator/SparseToDenseSparseToDense/topic_indicator/to_sparse_input/indices:index:04topic_indicator/to_sparse_input/dense_shape:output:0<topic_indicator/hash_table_Lookup/LookupTableFindV2:values:04topic_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????b
topic_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
topic_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    _
topic_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
topic_indicator/one_hotOneHot%topic_indicator/SparseToDense:dense:0&topic_indicator/one_hot/depth:output:0&topic_indicator/one_hot/Const:output:0(topic_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????x
%topic_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
topic_indicator/SumSum topic_indicator/one_hot:output:0.topic_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????a
topic_indicator/ShapeShapetopic_indicator/Sum:output:0*
T0*
_output_shapes
:m
#topic_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%topic_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%topic_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
topic_indicator/strided_sliceStridedSlicetopic_indicator/Shape:output:0,topic_indicator/strided_slice/stack:output:0.topic_indicator/strided_slice/stack_1:output:0.topic_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
topic_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
topic_indicator/Reshape/shapePack&topic_indicator/strided_slice:output:0(topic_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
topic_indicator/ReshapeReshapetopic_indicator/Sum:output:0&topic_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2age_bucketized/Reshape:output:0"emotion_indicator/Reshape:output:0!gender_indicator/Reshape:output:0&nationality_indicator/Reshape:output:0%occupation_indicator/Reshape:output:0 topic_indicator/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp6^emotion_indicator/hash_table_Lookup/LookupTableFindV25^gender_indicator/hash_table_Lookup/LookupTableFindV2:^nationality_indicator/hash_table_Lookup/LookupTableFindV29^occupation_indicator/hash_table_Lookup/LookupTableFindV24^topic_indicator/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2n
5emotion_indicator/hash_table_Lookup/LookupTableFindV25emotion_indicator/hash_table_Lookup/LookupTableFindV22l
4gender_indicator/hash_table_Lookup/LookupTableFindV24gender_indicator/hash_table_Lookup/LookupTableFindV22v
9nationality_indicator/hash_table_Lookup/LookupTableFindV29nationality_indicator/hash_table_Lookup/LookupTableFindV22t
8occupation_indicator/hash_table_Lookup/LookupTableFindV28occupation_indicator/hash_table_Lookup/LookupTableFindV22j
3topic_indicator/hash_table_Lookup/LookupTableFindV23topic_indicator/hash_table_Lookup/LookupTableFindV2:Q M
#
_output_shapes
:?????????
&
_user_specified_namefeatures/age:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/emotion:TP
#
_output_shapes
:?????????
)
_user_specified_namefeatures/gender:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/nationality:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/occupation:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/topic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
-
__inference__destroyer_466562
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
-
__inference__destroyer_466634
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
;
__inference__creator_466567
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name162*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
;
__inference__creator_466549
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name126*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?

?
D__inference_dense_15_layer_call_and_return_conditional_losses_466544

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_465470
age	
emotion

gender
nationality

occupation	
topic
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9:
??

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallageemotiongendernationality
occupationtopicunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*#
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_465385o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:?????????

_user_specified_nameage:LH
#
_output_shapes
:?????????
!
_user_specified_name	emotion:KG
#
_output_shapes
:?????????
 
_user_specified_namegender:PL
#
_output_shapes
:?????????
%
_user_specified_namenationality:OK
#
_output_shapes
:?????????
$
_user_specified_name
occupation:JF
#
_output_shapes
:?????????

_user_specified_nametopic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?	
J__inference_dense_features_layer_call_and_return_conditional_losses_464893
features	

features_1

features_2

features_3

features_4

features_5F
Bemotion_indicator_hash_table_lookup_lookuptablefindv2_table_handleG
Cemotion_indicator_hash_table_lookup_lookuptablefindv2_default_value	E
Agender_indicator_hash_table_lookup_lookuptablefindv2_table_handleF
Bgender_indicator_hash_table_lookup_lookuptablefindv2_default_value	J
Fnationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleK
Gnationality_indicator_hash_table_lookup_lookuptablefindv2_default_value	I
Eoccupation_indicator_hash_table_lookup_lookuptablefindv2_table_handleJ
Foccupation_indicator_hash_table_lookup_lookuptablefindv2_default_value	D
@topic_indicator_hash_table_lookup_lookuptablefindv2_table_handleE
Atopic_indicator_hash_table_lookup_lookuptablefindv2_default_value	
identity??5emotion_indicator/hash_table_Lookup/LookupTableFindV2?4gender_indicator/hash_table_Lookup/LookupTableFindV2?9nationality_indicator/hash_table_Lookup/LookupTableFindV2?8occupation_indicator/hash_table_Lookup/LookupTableFindV2?3topic_indicator/hash_table_Lookup/LookupTableFindV2h
age_bucketized/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
age_bucketized/ExpandDims
ExpandDimsfeatures&age_bucketized/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
age_bucketized/CastCast"age_bucketized/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
age_bucketized/Bucketize	Bucketizeage_bucketized/Cast:y:0*
T0*'
_output_shapes
:?????????*^

boundariesP
N"L  ?@   A  pA  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B  ?B  ?B  ?B  ?B  ?B  ?B?
age_bucketized/Cast_1Cast!age_bucketized/Bucketize:output:0*

DstT0	*

SrcT0*'
_output_shapes
:?????????a
age_bucketized/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??c
age_bucketized/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ^
age_bucketized/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
age_bucketized/one_hotOneHotage_bucketized/Cast_1:y:0%age_bucketized/one_hot/depth:output:0%age_bucketized/one_hot/Const:output:0'age_bucketized/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????c
age_bucketized/ShapeShapeage_bucketized/one_hot:output:0*
T0*
_output_shapes
:l
"age_bucketized/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$age_bucketized/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$age_bucketized/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
age_bucketized/strided_sliceStridedSliceage_bucketized/Shape:output:0+age_bucketized/strided_slice/stack:output:0-age_bucketized/strided_slice/stack_1:output:0-age_bucketized/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
age_bucketized/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
age_bucketized/Reshape/shapePack%age_bucketized/strided_slice:output:0'age_bucketized/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
age_bucketized/ReshapeReshapeage_bucketized/one_hot:output:0%age_bucketized/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 emotion_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
emotion_indicator/ExpandDims
ExpandDims
features_1)emotion_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????q
0emotion_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
*emotion_indicator/to_sparse_input/NotEqualNotEqual%emotion_indicator/ExpandDims:output:09emotion_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
)emotion_indicator/to_sparse_input/indicesWhere.emotion_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(emotion_indicator/to_sparse_input/valuesGatherNd%emotion_indicator/ExpandDims:output:01emotion_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
-emotion_indicator/to_sparse_input/dense_shapeShape%emotion_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
5emotion_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Bemotion_indicator_hash_table_lookup_lookuptablefindv2_table_handle1emotion_indicator/to_sparse_input/values:output:0Cemotion_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????x
-emotion_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
emotion_indicator/SparseToDenseSparseToDense1emotion_indicator/to_sparse_input/indices:index:06emotion_indicator/to_sparse_input/dense_shape:output:0>emotion_indicator/hash_table_Lookup/LookupTableFindV2:values:06emotion_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????d
emotion_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??f
!emotion_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    a
emotion_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
emotion_indicator/one_hotOneHot'emotion_indicator/SparseToDense:dense:0(emotion_indicator/one_hot/depth:output:0(emotion_indicator/one_hot/Const:output:0*emotion_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????z
'emotion_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
emotion_indicator/SumSum"emotion_indicator/one_hot:output:00emotion_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????e
emotion_indicator/ShapeShapeemotion_indicator/Sum:output:0*
T0*
_output_shapes
:o
%emotion_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'emotion_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'emotion_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
emotion_indicator/strided_sliceStridedSlice emotion_indicator/Shape:output:0.emotion_indicator/strided_slice/stack:output:00emotion_indicator/strided_slice/stack_1:output:00emotion_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!emotion_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
emotion_indicator/Reshape/shapePack(emotion_indicator/strided_slice:output:0*emotion_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
emotion_indicator/ReshapeReshapeemotion_indicator/Sum:output:0(emotion_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
gender_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gender_indicator/ExpandDims
ExpandDims
features_2(gender_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????p
/gender_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
)gender_indicator/to_sparse_input/NotEqualNotEqual$gender_indicator/ExpandDims:output:08gender_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
(gender_indicator/to_sparse_input/indicesWhere-gender_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
'gender_indicator/to_sparse_input/valuesGatherNd$gender_indicator/ExpandDims:output:00gender_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
,gender_indicator/to_sparse_input/dense_shapeShape$gender_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
4gender_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Agender_indicator_hash_table_lookup_lookuptablefindv2_table_handle0gender_indicator/to_sparse_input/values:output:0Bgender_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????w
,gender_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
gender_indicator/SparseToDenseSparseToDense0gender_indicator/to_sparse_input/indices:index:05gender_indicator/to_sparse_input/dense_shape:output:0=gender_indicator/hash_table_Lookup/LookupTableFindV2:values:05gender_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????c
gender_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
 gender_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    `
gender_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
gender_indicator/one_hotOneHot&gender_indicator/SparseToDense:dense:0'gender_indicator/one_hot/depth:output:0'gender_indicator/one_hot/Const:output:0)gender_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????y
&gender_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
gender_indicator/SumSum!gender_indicator/one_hot:output:0/gender_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????c
gender_indicator/ShapeShapegender_indicator/Sum:output:0*
T0*
_output_shapes
:n
$gender_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&gender_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&gender_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gender_indicator/strided_sliceStridedSlicegender_indicator/Shape:output:0-gender_indicator/strided_slice/stack:output:0/gender_indicator/strided_slice/stack_1:output:0/gender_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 gender_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
gender_indicator/Reshape/shapePack'gender_indicator/strided_slice:output:0)gender_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
gender_indicator/ReshapeReshapegender_indicator/Sum:output:0'gender_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$nationality_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 nationality_indicator/ExpandDims
ExpandDims
features_3-nationality_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4nationality_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.nationality_indicator/to_sparse_input/NotEqualNotEqual)nationality_indicator/ExpandDims:output:0=nationality_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-nationality_indicator/to_sparse_input/indicesWhere2nationality_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,nationality_indicator/to_sparse_input/valuesGatherNd)nationality_indicator/ExpandDims:output:05nationality_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1nationality_indicator/to_sparse_input/dense_shapeShape)nationality_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
9nationality_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Fnationality_indicator_hash_table_lookup_lookuptablefindv2_table_handle5nationality_indicator/to_sparse_input/values:output:0Gnationality_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1nationality_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#nationality_indicator/SparseToDenseSparseToDense5nationality_indicator/to_sparse_input/indices:index:0:nationality_indicator/to_sparse_input/dense_shape:output:0Bnationality_indicator/hash_table_Lookup/LookupTableFindV2:values:0:nationality_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#nationality_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%nationality_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#nationality_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
nationality_indicator/one_hotOneHot+nationality_indicator/SparseToDense:dense:0,nationality_indicator/one_hot/depth:output:0,nationality_indicator/one_hot/Const:output:0.nationality_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+nationality_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
nationality_indicator/SumSum&nationality_indicator/one_hot:output:04nationality_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
nationality_indicator/ShapeShape"nationality_indicator/Sum:output:0*
T0*
_output_shapes
:s
)nationality_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+nationality_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+nationality_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#nationality_indicator/strided_sliceStridedSlice$nationality_indicator/Shape:output:02nationality_indicator/strided_slice/stack:output:04nationality_indicator/strided_slice/stack_1:output:04nationality_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%nationality_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#nationality_indicator/Reshape/shapePack,nationality_indicator/strided_slice:output:0.nationality_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
nationality_indicator/ReshapeReshape"nationality_indicator/Sum:output:0,nationality_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????n
#occupation_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
occupation_indicator/ExpandDims
ExpandDims
features_4,occupation_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3occupation_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-occupation_indicator/to_sparse_input/NotEqualNotEqual(occupation_indicator/ExpandDims:output:0<occupation_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,occupation_indicator/to_sparse_input/indicesWhere1occupation_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+occupation_indicator/to_sparse_input/valuesGatherNd(occupation_indicator/ExpandDims:output:04occupation_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0occupation_indicator/to_sparse_input/dense_shapeShape(occupation_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
8occupation_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Eoccupation_indicator_hash_table_lookup_lookuptablefindv2_table_handle4occupation_indicator/to_sparse_input/values:output:0Foccupation_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0occupation_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"occupation_indicator/SparseToDenseSparseToDense4occupation_indicator/to_sparse_input/indices:index:09occupation_indicator/to_sparse_input/dense_shape:output:0Aoccupation_indicator/hash_table_Lookup/LookupTableFindV2:values:09occupation_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"occupation_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$occupation_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"occupation_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
occupation_indicator/one_hotOneHot*occupation_indicator/SparseToDense:dense:0+occupation_indicator/one_hot/depth:output:0+occupation_indicator/one_hot/Const:output:0-occupation_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*occupation_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
occupation_indicator/SumSum%occupation_indicator/one_hot:output:03occupation_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
occupation_indicator/ShapeShape!occupation_indicator/Sum:output:0*
T0*
_output_shapes
:r
(occupation_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*occupation_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*occupation_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"occupation_indicator/strided_sliceStridedSlice#occupation_indicator/Shape:output:01occupation_indicator/strided_slice/stack:output:03occupation_indicator/strided_slice/stack_1:output:03occupation_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$occupation_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"occupation_indicator/Reshape/shapePack+occupation_indicator/strided_slice:output:0-occupation_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
occupation_indicator/ReshapeReshape!occupation_indicator/Sum:output:0+occupation_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
topic_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
topic_indicator/ExpandDims
ExpandDims
features_5'topic_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????o
.topic_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
(topic_indicator/to_sparse_input/NotEqualNotEqual#topic_indicator/ExpandDims:output:07topic_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
'topic_indicator/to_sparse_input/indicesWhere,topic_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
&topic_indicator/to_sparse_input/valuesGatherNd#topic_indicator/ExpandDims:output:0/topic_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
+topic_indicator/to_sparse_input/dense_shapeShape#topic_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3topic_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@topic_indicator_hash_table_lookup_lookuptablefindv2_table_handle/topic_indicator/to_sparse_input/values:output:0Atopic_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????v
+topic_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
topic_indicator/SparseToDenseSparseToDense/topic_indicator/to_sparse_input/indices:index:04topic_indicator/to_sparse_input/dense_shape:output:0<topic_indicator/hash_table_Lookup/LookupTableFindV2:values:04topic_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????b
topic_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
topic_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    _
topic_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
topic_indicator/one_hotOneHot%topic_indicator/SparseToDense:dense:0&topic_indicator/one_hot/depth:output:0&topic_indicator/one_hot/Const:output:0(topic_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????x
%topic_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
topic_indicator/SumSum topic_indicator/one_hot:output:0.topic_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????a
topic_indicator/ShapeShapetopic_indicator/Sum:output:0*
T0*
_output_shapes
:m
#topic_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%topic_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%topic_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
topic_indicator/strided_sliceStridedSlicetopic_indicator/Shape:output:0,topic_indicator/strided_slice/stack:output:0.topic_indicator/strided_slice/stack_1:output:0.topic_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
topic_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
topic_indicator/Reshape/shapePack&topic_indicator/strided_slice:output:0(topic_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
topic_indicator/ReshapeReshapetopic_indicator/Sum:output:0&topic_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2age_bucketized/Reshape:output:0"emotion_indicator/Reshape:output:0!gender_indicator/Reshape:output:0&nationality_indicator/Reshape:output:0%occupation_indicator/Reshape:output:0 topic_indicator/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp6^emotion_indicator/hash_table_Lookup/LookupTableFindV25^gender_indicator/hash_table_Lookup/LookupTableFindV2:^nationality_indicator/hash_table_Lookup/LookupTableFindV29^occupation_indicator/hash_table_Lookup/LookupTableFindV24^topic_indicator/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2n
5emotion_indicator/hash_table_Lookup/LookupTableFindV25emotion_indicator/hash_table_Lookup/LookupTableFindV22l
4gender_indicator/hash_table_Lookup/LookupTableFindV24gender_indicator/hash_table_Lookup/LookupTableFindV22v
9nationality_indicator/hash_table_Lookup/LookupTableFindV29nationality_indicator/hash_table_Lookup/LookupTableFindV22t
8occupation_indicator/hash_table_Lookup/LookupTableFindV28occupation_indicator/hash_table_Lookup/LookupTableFindV22j
3topic_indicator/hash_table_Lookup/LookupTableFindV23topic_indicator/hash_table_Lookup/LookupTableFindV2:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?%
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_465570
age	
emotion

gender
nationality

occupation	
topic
dense_features_465528
dense_features_465530	
dense_features_465532
dense_features_465534	
dense_features_465536
dense_features_465538	
dense_features_465540
dense_features_465542	
dense_features_465544
dense_features_465546	#
dense_12_465549:
??
dense_12_465551:	?#
dense_13_465554:
??
dense_13_465556:	?#
dense_14_465559:
??
dense_14_465561:	?"
dense_15_465564:	?
dense_15_465566:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallageemotiongendernationality
occupationtopicdense_features_465528dense_features_465530dense_features_465532dense_features_465534dense_features_465536dense_features_465538dense_features_465540dense_features_465542dense_features_465544dense_features_465546*
Tin
2						*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_465259?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_12_465549dense_12_465551*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_464926?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_465554dense_13_465556*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_464943?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_465559dense_14_465561*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_464960?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_465564dense_15_465566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_464977x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:H D
#
_output_shapes
:?????????

_user_specified_nameage:LH
#
_output_shapes
:?????????
!
_user_specified_name	emotion:KG
#
_output_shapes
:?????????
 
_user_specified_namegender:PL
#
_output_shapes
:?????????
%
_user_specified_namenationality:OK
#
_output_shapes
:?????????
$
_user_specified_name
occupation:JF
#
_output_shapes
:?????????

_user_specified_nametopic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ּ
?
!__inference__wrapped_model_464718
age	
emotion

gender
nationality

occupation	
topicb
^sequential_3_dense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_table_handlec
_sequential_3_dense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_default_value	a
]sequential_3_dense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_table_handleb
^sequential_3_dense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_default_value	f
bsequential_3_dense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleg
csequential_3_dense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_default_value	e
asequential_3_dense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_table_handlef
bsequential_3_dense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_default_value	`
\sequential_3_dense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_table_handlea
]sequential_3_dense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_default_value	H
4sequential_3_dense_12_matmul_readvariableop_resource:
??D
5sequential_3_dense_12_biasadd_readvariableop_resource:	?H
4sequential_3_dense_13_matmul_readvariableop_resource:
??D
5sequential_3_dense_13_biasadd_readvariableop_resource:	?H
4sequential_3_dense_14_matmul_readvariableop_resource:
??D
5sequential_3_dense_14_biasadd_readvariableop_resource:	?G
4sequential_3_dense_15_matmul_readvariableop_resource:	?C
5sequential_3_dense_15_biasadd_readvariableop_resource:
identity??,sequential_3/dense_12/BiasAdd/ReadVariableOp?+sequential_3/dense_12/MatMul/ReadVariableOp?,sequential_3/dense_13/BiasAdd/ReadVariableOp?+sequential_3/dense_13/MatMul/ReadVariableOp?,sequential_3/dense_14/BiasAdd/ReadVariableOp?+sequential_3/dense_14/MatMul/ReadVariableOp?,sequential_3/dense_15/BiasAdd/ReadVariableOp?+sequential_3/dense_15/MatMul/ReadVariableOp?Qsequential_3/dense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2?Psequential_3/dense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2?Usequential_3/dense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2?Tsequential_3/dense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2?Osequential_3/dense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2?
9sequential_3/dense_features/age_bucketized/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5sequential_3/dense_features/age_bucketized/ExpandDims
ExpandDimsageBsequential_3/dense_features/age_bucketized/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
/sequential_3/dense_features/age_bucketized/CastCast>sequential_3/dense_features/age_bucketized/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
4sequential_3/dense_features/age_bucketized/Bucketize	Bucketize3sequential_3/dense_features/age_bucketized/Cast:y:0*
T0*'
_output_shapes
:?????????*^

boundariesP
N"L  ?@   A  pA  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B  ?B  ?B  ?B  ?B  ?B  ?B?
1sequential_3/dense_features/age_bucketized/Cast_1Cast=sequential_3/dense_features/age_bucketized/Bucketize:output:0*

DstT0	*

SrcT0*'
_output_shapes
:?????????}
8sequential_3/dense_features/age_bucketized/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
:sequential_3/dense_features/age_bucketized/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    z
8sequential_3/dense_features/age_bucketized/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
2sequential_3/dense_features/age_bucketized/one_hotOneHot5sequential_3/dense_features/age_bucketized/Cast_1:y:0Asequential_3/dense_features/age_bucketized/one_hot/depth:output:0Asequential_3/dense_features/age_bucketized/one_hot/Const:output:0Csequential_3/dense_features/age_bucketized/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
0sequential_3/dense_features/age_bucketized/ShapeShape;sequential_3/dense_features/age_bucketized/one_hot:output:0*
T0*
_output_shapes
:?
>sequential_3/dense_features/age_bucketized/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@sequential_3/dense_features/age_bucketized/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential_3/dense_features/age_bucketized/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_3/dense_features/age_bucketized/strided_sliceStridedSlice9sequential_3/dense_features/age_bucketized/Shape:output:0Gsequential_3/dense_features/age_bucketized/strided_slice/stack:output:0Isequential_3/dense_features/age_bucketized/strided_slice/stack_1:output:0Isequential_3/dense_features/age_bucketized/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:sequential_3/dense_features/age_bucketized/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8sequential_3/dense_features/age_bucketized/Reshape/shapePackAsequential_3/dense_features/age_bucketized/strided_slice:output:0Csequential_3/dense_features/age_bucketized/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
2sequential_3/dense_features/age_bucketized/ReshapeReshape;sequential_3/dense_features/age_bucketized/one_hot:output:0Asequential_3/dense_features/age_bucketized/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
<sequential_3/dense_features/emotion_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
8sequential_3/dense_features/emotion_indicator/ExpandDims
ExpandDimsemotionEsequential_3/dense_features/emotion_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Lsequential_3/dense_features/emotion_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Fsequential_3/dense_features/emotion_indicator/to_sparse_input/NotEqualNotEqualAsequential_3/dense_features/emotion_indicator/ExpandDims:output:0Usequential_3/dense_features/emotion_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Esequential_3/dense_features/emotion_indicator/to_sparse_input/indicesWhereJsequential_3/dense_features/emotion_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Dsequential_3/dense_features/emotion_indicator/to_sparse_input/valuesGatherNdAsequential_3/dense_features/emotion_indicator/ExpandDims:output:0Msequential_3/dense_features/emotion_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Isequential_3/dense_features/emotion_indicator/to_sparse_input/dense_shapeShapeAsequential_3/dense_features/emotion_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Qsequential_3/dense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2^sequential_3_dense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_table_handleMsequential_3/dense_features/emotion_indicator/to_sparse_input/values:output:0_sequential_3_dense_features_emotion_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Isequential_3/dense_features/emotion_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
;sequential_3/dense_features/emotion_indicator/SparseToDenseSparseToDenseMsequential_3/dense_features/emotion_indicator/to_sparse_input/indices:index:0Rsequential_3/dense_features/emotion_indicator/to_sparse_input/dense_shape:output:0Zsequential_3/dense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2:values:0Rsequential_3/dense_features/emotion_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
;sequential_3/dense_features/emotion_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
=sequential_3/dense_features/emotion_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    }
;sequential_3/dense_features/emotion_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
5sequential_3/dense_features/emotion_indicator/one_hotOneHotCsequential_3/dense_features/emotion_indicator/SparseToDense:dense:0Dsequential_3/dense_features/emotion_indicator/one_hot/depth:output:0Dsequential_3/dense_features/emotion_indicator/one_hot/Const:output:0Fsequential_3/dense_features/emotion_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Csequential_3/dense_features/emotion_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
1sequential_3/dense_features/emotion_indicator/SumSum>sequential_3/dense_features/emotion_indicator/one_hot:output:0Lsequential_3/dense_features/emotion_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
3sequential_3/dense_features/emotion_indicator/ShapeShape:sequential_3/dense_features/emotion_indicator/Sum:output:0*
T0*
_output_shapes
:?
Asequential_3/dense_features/emotion_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csequential_3/dense_features/emotion_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csequential_3/dense_features/emotion_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential_3/dense_features/emotion_indicator/strided_sliceStridedSlice<sequential_3/dense_features/emotion_indicator/Shape:output:0Jsequential_3/dense_features/emotion_indicator/strided_slice/stack:output:0Lsequential_3/dense_features/emotion_indicator/strided_slice/stack_1:output:0Lsequential_3/dense_features/emotion_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
=sequential_3/dense_features/emotion_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
;sequential_3/dense_features/emotion_indicator/Reshape/shapePackDsequential_3/dense_features/emotion_indicator/strided_slice:output:0Fsequential_3/dense_features/emotion_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
5sequential_3/dense_features/emotion_indicator/ReshapeReshape:sequential_3/dense_features/emotion_indicator/Sum:output:0Dsequential_3/dense_features/emotion_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
;sequential_3/dense_features/gender_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
7sequential_3/dense_features/gender_indicator/ExpandDims
ExpandDimsgenderDsequential_3/dense_features/gender_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Ksequential_3/dense_features/gender_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Esequential_3/dense_features/gender_indicator/to_sparse_input/NotEqualNotEqual@sequential_3/dense_features/gender_indicator/ExpandDims:output:0Tsequential_3/dense_features/gender_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Dsequential_3/dense_features/gender_indicator/to_sparse_input/indicesWhereIsequential_3/dense_features/gender_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Csequential_3/dense_features/gender_indicator/to_sparse_input/valuesGatherNd@sequential_3/dense_features/gender_indicator/ExpandDims:output:0Lsequential_3/dense_features/gender_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Hsequential_3/dense_features/gender_indicator/to_sparse_input/dense_shapeShape@sequential_3/dense_features/gender_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Psequential_3/dense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2]sequential_3_dense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_table_handleLsequential_3/dense_features/gender_indicator/to_sparse_input/values:output:0^sequential_3_dense_features_gender_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Hsequential_3/dense_features/gender_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
:sequential_3/dense_features/gender_indicator/SparseToDenseSparseToDenseLsequential_3/dense_features/gender_indicator/to_sparse_input/indices:index:0Qsequential_3/dense_features/gender_indicator/to_sparse_input/dense_shape:output:0Ysequential_3/dense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2:values:0Qsequential_3/dense_features/gender_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????
:sequential_3/dense_features/gender_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
<sequential_3/dense_features/gender_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    |
:sequential_3/dense_features/gender_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
4sequential_3/dense_features/gender_indicator/one_hotOneHotBsequential_3/dense_features/gender_indicator/SparseToDense:dense:0Csequential_3/dense_features/gender_indicator/one_hot/depth:output:0Csequential_3/dense_features/gender_indicator/one_hot/Const:output:0Esequential_3/dense_features/gender_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Bsequential_3/dense_features/gender_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
0sequential_3/dense_features/gender_indicator/SumSum=sequential_3/dense_features/gender_indicator/one_hot:output:0Ksequential_3/dense_features/gender_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
2sequential_3/dense_features/gender_indicator/ShapeShape9sequential_3/dense_features/gender_indicator/Sum:output:0*
T0*
_output_shapes
:?
@sequential_3/dense_features/gender_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential_3/dense_features/gender_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_3/dense_features/gender_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential_3/dense_features/gender_indicator/strided_sliceStridedSlice;sequential_3/dense_features/gender_indicator/Shape:output:0Isequential_3/dense_features/gender_indicator/strided_slice/stack:output:0Ksequential_3/dense_features/gender_indicator/strided_slice/stack_1:output:0Ksequential_3/dense_features/gender_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<sequential_3/dense_features/gender_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
:sequential_3/dense_features/gender_indicator/Reshape/shapePackCsequential_3/dense_features/gender_indicator/strided_slice:output:0Esequential_3/dense_features/gender_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
4sequential_3/dense_features/gender_indicator/ReshapeReshape9sequential_3/dense_features/gender_indicator/Sum:output:0Csequential_3/dense_features/gender_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
@sequential_3/dense_features/nationality_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_3/dense_features/nationality_indicator/ExpandDims
ExpandDimsnationalityIsequential_3/dense_features/nationality_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Psequential_3/dense_features/nationality_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Jsequential_3/dense_features/nationality_indicator/to_sparse_input/NotEqualNotEqualEsequential_3/dense_features/nationality_indicator/ExpandDims:output:0Ysequential_3/dense_features/nationality_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Isequential_3/dense_features/nationality_indicator/to_sparse_input/indicesWhereNsequential_3/dense_features/nationality_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Hsequential_3/dense_features/nationality_indicator/to_sparse_input/valuesGatherNdEsequential_3/dense_features/nationality_indicator/ExpandDims:output:0Qsequential_3/dense_features/nationality_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Msequential_3/dense_features/nationality_indicator/to_sparse_input/dense_shapeShapeEsequential_3/dense_features/nationality_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Usequential_3/dense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2bsequential_3_dense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleQsequential_3/dense_features/nationality_indicator/to_sparse_input/values:output:0csequential_3_dense_features_nationality_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Msequential_3/dense_features/nationality_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
?sequential_3/dense_features/nationality_indicator/SparseToDenseSparseToDenseQsequential_3/dense_features/nationality_indicator/to_sparse_input/indices:index:0Vsequential_3/dense_features/nationality_indicator/to_sparse_input/dense_shape:output:0^sequential_3/dense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2:values:0Vsequential_3/dense_features/nationality_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
?sequential_3/dense_features/nationality_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Asequential_3/dense_features/nationality_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
?sequential_3/dense_features/nationality_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
9sequential_3/dense_features/nationality_indicator/one_hotOneHotGsequential_3/dense_features/nationality_indicator/SparseToDense:dense:0Hsequential_3/dense_features/nationality_indicator/one_hot/depth:output:0Hsequential_3/dense_features/nationality_indicator/one_hot/Const:output:0Jsequential_3/dense_features/nationality_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
Gsequential_3/dense_features/nationality_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
5sequential_3/dense_features/nationality_indicator/SumSumBsequential_3/dense_features/nationality_indicator/one_hot:output:0Psequential_3/dense_features/nationality_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
7sequential_3/dense_features/nationality_indicator/ShapeShape>sequential_3/dense_features/nationality_indicator/Sum:output:0*
T0*
_output_shapes
:?
Esequential_3/dense_features/nationality_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential_3/dense_features/nationality_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Gsequential_3/dense_features/nationality_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential_3/dense_features/nationality_indicator/strided_sliceStridedSlice@sequential_3/dense_features/nationality_indicator/Shape:output:0Nsequential_3/dense_features/nationality_indicator/strided_slice/stack:output:0Psequential_3/dense_features/nationality_indicator/strided_slice/stack_1:output:0Psequential_3/dense_features/nationality_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Asequential_3/dense_features/nationality_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
?sequential_3/dense_features/nationality_indicator/Reshape/shapePackHsequential_3/dense_features/nationality_indicator/strided_slice:output:0Jsequential_3/dense_features/nationality_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
9sequential_3/dense_features/nationality_indicator/ReshapeReshape>sequential_3/dense_features/nationality_indicator/Sum:output:0Hsequential_3/dense_features/nationality_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
?sequential_3/dense_features/occupation_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
;sequential_3/dense_features/occupation_indicator/ExpandDims
ExpandDims
occupationHsequential_3/dense_features/occupation_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Osequential_3/dense_features/occupation_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Isequential_3/dense_features/occupation_indicator/to_sparse_input/NotEqualNotEqualDsequential_3/dense_features/occupation_indicator/ExpandDims:output:0Xsequential_3/dense_features/occupation_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Hsequential_3/dense_features/occupation_indicator/to_sparse_input/indicesWhereMsequential_3/dense_features/occupation_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Gsequential_3/dense_features/occupation_indicator/to_sparse_input/valuesGatherNdDsequential_3/dense_features/occupation_indicator/ExpandDims:output:0Psequential_3/dense_features/occupation_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Lsequential_3/dense_features/occupation_indicator/to_sparse_input/dense_shapeShapeDsequential_3/dense_features/occupation_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Tsequential_3/dense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2asequential_3_dense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_table_handlePsequential_3/dense_features/occupation_indicator/to_sparse_input/values:output:0bsequential_3_dense_features_occupation_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Lsequential_3/dense_features/occupation_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
>sequential_3/dense_features/occupation_indicator/SparseToDenseSparseToDensePsequential_3/dense_features/occupation_indicator/to_sparse_input/indices:index:0Usequential_3/dense_features/occupation_indicator/to_sparse_input/dense_shape:output:0]sequential_3/dense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2:values:0Usequential_3/dense_features/occupation_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
>sequential_3/dense_features/occupation_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
@sequential_3/dense_features/occupation_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
>sequential_3/dense_features/occupation_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
8sequential_3/dense_features/occupation_indicator/one_hotOneHotFsequential_3/dense_features/occupation_indicator/SparseToDense:dense:0Gsequential_3/dense_features/occupation_indicator/one_hot/depth:output:0Gsequential_3/dense_features/occupation_indicator/one_hot/Const:output:0Isequential_3/dense_features/occupation_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Fsequential_3/dense_features/occupation_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
4sequential_3/dense_features/occupation_indicator/SumSumAsequential_3/dense_features/occupation_indicator/one_hot:output:0Osequential_3/dense_features/occupation_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
6sequential_3/dense_features/occupation_indicator/ShapeShape=sequential_3/dense_features/occupation_indicator/Sum:output:0*
T0*
_output_shapes
:?
Dsequential_3/dense_features/occupation_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential_3/dense_features/occupation_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_3/dense_features/occupation_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential_3/dense_features/occupation_indicator/strided_sliceStridedSlice?sequential_3/dense_features/occupation_indicator/Shape:output:0Msequential_3/dense_features/occupation_indicator/strided_slice/stack:output:0Osequential_3/dense_features/occupation_indicator/strided_slice/stack_1:output:0Osequential_3/dense_features/occupation_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_3/dense_features/occupation_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
>sequential_3/dense_features/occupation_indicator/Reshape/shapePackGsequential_3/dense_features/occupation_indicator/strided_slice:output:0Isequential_3/dense_features/occupation_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
8sequential_3/dense_features/occupation_indicator/ReshapeReshape=sequential_3/dense_features/occupation_indicator/Sum:output:0Gsequential_3/dense_features/occupation_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
:sequential_3/dense_features/topic_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
6sequential_3/dense_features/topic_indicator/ExpandDims
ExpandDimstopicCsequential_3/dense_features/topic_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Jsequential_3/dense_features/topic_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Dsequential_3/dense_features/topic_indicator/to_sparse_input/NotEqualNotEqual?sequential_3/dense_features/topic_indicator/ExpandDims:output:0Ssequential_3/dense_features/topic_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Csequential_3/dense_features/topic_indicator/to_sparse_input/indicesWhereHsequential_3/dense_features/topic_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Bsequential_3/dense_features/topic_indicator/to_sparse_input/valuesGatherNd?sequential_3/dense_features/topic_indicator/ExpandDims:output:0Ksequential_3/dense_features/topic_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Gsequential_3/dense_features/topic_indicator/to_sparse_input/dense_shapeShape?sequential_3/dense_features/topic_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Osequential_3/dense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2\sequential_3_dense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_table_handleKsequential_3/dense_features/topic_indicator/to_sparse_input/values:output:0]sequential_3_dense_features_topic_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Gsequential_3/dense_features/topic_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
9sequential_3/dense_features/topic_indicator/SparseToDenseSparseToDenseKsequential_3/dense_features/topic_indicator/to_sparse_input/indices:index:0Psequential_3/dense_features/topic_indicator/to_sparse_input/dense_shape:output:0Xsequential_3/dense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2:values:0Psequential_3/dense_features/topic_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????~
9sequential_3/dense_features/topic_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;sequential_3/dense_features/topic_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    {
9sequential_3/dense_features/topic_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
3sequential_3/dense_features/topic_indicator/one_hotOneHotAsequential_3/dense_features/topic_indicator/SparseToDense:dense:0Bsequential_3/dense_features/topic_indicator/one_hot/depth:output:0Bsequential_3/dense_features/topic_indicator/one_hot/Const:output:0Dsequential_3/dense_features/topic_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Asequential_3/dense_features/topic_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/sequential_3/dense_features/topic_indicator/SumSum<sequential_3/dense_features/topic_indicator/one_hot:output:0Jsequential_3/dense_features/topic_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
1sequential_3/dense_features/topic_indicator/ShapeShape8sequential_3/dense_features/topic_indicator/Sum:output:0*
T0*
_output_shapes
:?
?sequential_3/dense_features/topic_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential_3/dense_features/topic_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential_3/dense_features/topic_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential_3/dense_features/topic_indicator/strided_sliceStridedSlice:sequential_3/dense_features/topic_indicator/Shape:output:0Hsequential_3/dense_features/topic_indicator/strided_slice/stack:output:0Jsequential_3/dense_features/topic_indicator/strided_slice/stack_1:output:0Jsequential_3/dense_features/topic_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential_3/dense_features/topic_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
9sequential_3/dense_features/topic_indicator/Reshape/shapePackBsequential_3/dense_features/topic_indicator/strided_slice:output:0Dsequential_3/dense_features/topic_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3sequential_3/dense_features/topic_indicator/ReshapeReshape8sequential_3/dense_features/topic_indicator/Sum:output:0Bsequential_3/dense_features/topic_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????r
'sequential_3/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"sequential_3/dense_features/concatConcatV2;sequential_3/dense_features/age_bucketized/Reshape:output:0>sequential_3/dense_features/emotion_indicator/Reshape:output:0=sequential_3/dense_features/gender_indicator/Reshape:output:0Bsequential_3/dense_features/nationality_indicator/Reshape:output:0Asequential_3/dense_features/occupation_indicator/Reshape:output:0<sequential_3/dense_features/topic_indicator/Reshape:output:00sequential_3/dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_3/dense_12/MatMulMatMul+sequential_3/dense_features/concat:output:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
sequential_3/dense_12/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_3/dense_13/MatMulMatMul(sequential_3/dense_12/Relu:activations:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
sequential_3/dense_13/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_3/dense_14/MatMulMatMul(sequential_3/dense_13/Relu:activations:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
sequential_3/dense_14/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_3/dense_15/MatMulMatMul(sequential_3/dense_14/Relu:activations:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_3/dense_15/SoftmaxSoftmax&sequential_3/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????v
IdentityIdentity'sequential_3/dense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOpR^sequential_3/dense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2Q^sequential_3/dense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2V^sequential_3/dense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2U^sequential_3/dense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2P^sequential_3/dense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp2?
Qsequential_3/dense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV2Qsequential_3/dense_features/emotion_indicator/hash_table_Lookup/LookupTableFindV22?
Psequential_3/dense_features/gender_indicator/hash_table_Lookup/LookupTableFindV2Psequential_3/dense_features/gender_indicator/hash_table_Lookup/LookupTableFindV22?
Usequential_3/dense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV2Usequential_3/dense_features/nationality_indicator/hash_table_Lookup/LookupTableFindV22?
Tsequential_3/dense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV2Tsequential_3/dense_features/occupation_indicator/hash_table_Lookup/LookupTableFindV22?
Osequential_3/dense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2Osequential_3/dense_features/topic_indicator/hash_table_Lookup/LookupTableFindV2:H D
#
_output_shapes
:?????????

_user_specified_nameage:LH
#
_output_shapes
:?????????
!
_user_specified_name	emotion:KG
#
_output_shapes
:?????????
 
_user_specified_namegender:PL
#
_output_shapes
:?????????
%
_user_specified_namenationality:OK
#
_output_shapes
:?????????
$
_user_specified_name
occupation:JF
#
_output_shapes
:?????????

_user_specified_nametopic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_dense_13_layer_call_and_return_conditional_losses_466504

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?	
J__inference_dense_features_layer_call_and_return_conditional_losses_466306
features_age	
features_emotion
features_gender
features_nationality
features_occupation
features_topicF
Bemotion_indicator_hash_table_lookup_lookuptablefindv2_table_handleG
Cemotion_indicator_hash_table_lookup_lookuptablefindv2_default_value	E
Agender_indicator_hash_table_lookup_lookuptablefindv2_table_handleF
Bgender_indicator_hash_table_lookup_lookuptablefindv2_default_value	J
Fnationality_indicator_hash_table_lookup_lookuptablefindv2_table_handleK
Gnationality_indicator_hash_table_lookup_lookuptablefindv2_default_value	I
Eoccupation_indicator_hash_table_lookup_lookuptablefindv2_table_handleJ
Foccupation_indicator_hash_table_lookup_lookuptablefindv2_default_value	D
@topic_indicator_hash_table_lookup_lookuptablefindv2_table_handleE
Atopic_indicator_hash_table_lookup_lookuptablefindv2_default_value	
identity??5emotion_indicator/hash_table_Lookup/LookupTableFindV2?4gender_indicator/hash_table_Lookup/LookupTableFindV2?9nationality_indicator/hash_table_Lookup/LookupTableFindV2?8occupation_indicator/hash_table_Lookup/LookupTableFindV2?3topic_indicator/hash_table_Lookup/LookupTableFindV2h
age_bucketized/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
age_bucketized/ExpandDims
ExpandDimsfeatures_age&age_bucketized/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
age_bucketized/CastCast"age_bucketized/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
age_bucketized/Bucketize	Bucketizeage_bucketized/Cast:y:0*
T0*'
_output_shapes
:?????????*^

boundariesP
N"L  ?@   A  pA  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B  ?B  ?B  ?B  ?B  ?B  ?B?
age_bucketized/Cast_1Cast!age_bucketized/Bucketize:output:0*

DstT0	*

SrcT0*'
_output_shapes
:?????????a
age_bucketized/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??c
age_bucketized/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ^
age_bucketized/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
age_bucketized/one_hotOneHotage_bucketized/Cast_1:y:0%age_bucketized/one_hot/depth:output:0%age_bucketized/one_hot/Const:output:0'age_bucketized/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????c
age_bucketized/ShapeShapeage_bucketized/one_hot:output:0*
T0*
_output_shapes
:l
"age_bucketized/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$age_bucketized/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$age_bucketized/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
age_bucketized/strided_sliceStridedSliceage_bucketized/Shape:output:0+age_bucketized/strided_slice/stack:output:0-age_bucketized/strided_slice/stack_1:output:0-age_bucketized/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
age_bucketized/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
age_bucketized/Reshape/shapePack%age_bucketized/strided_slice:output:0'age_bucketized/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
age_bucketized/ReshapeReshapeage_bucketized/one_hot:output:0%age_bucketized/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 emotion_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
emotion_indicator/ExpandDims
ExpandDimsfeatures_emotion)emotion_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????q
0emotion_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
*emotion_indicator/to_sparse_input/NotEqualNotEqual%emotion_indicator/ExpandDims:output:09emotion_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
)emotion_indicator/to_sparse_input/indicesWhere.emotion_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(emotion_indicator/to_sparse_input/valuesGatherNd%emotion_indicator/ExpandDims:output:01emotion_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
-emotion_indicator/to_sparse_input/dense_shapeShape%emotion_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
5emotion_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Bemotion_indicator_hash_table_lookup_lookuptablefindv2_table_handle1emotion_indicator/to_sparse_input/values:output:0Cemotion_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????x
-emotion_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
emotion_indicator/SparseToDenseSparseToDense1emotion_indicator/to_sparse_input/indices:index:06emotion_indicator/to_sparse_input/dense_shape:output:0>emotion_indicator/hash_table_Lookup/LookupTableFindV2:values:06emotion_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????d
emotion_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??f
!emotion_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    a
emotion_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
emotion_indicator/one_hotOneHot'emotion_indicator/SparseToDense:dense:0(emotion_indicator/one_hot/depth:output:0(emotion_indicator/one_hot/Const:output:0*emotion_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????z
'emotion_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
emotion_indicator/SumSum"emotion_indicator/one_hot:output:00emotion_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????e
emotion_indicator/ShapeShapeemotion_indicator/Sum:output:0*
T0*
_output_shapes
:o
%emotion_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'emotion_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'emotion_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
emotion_indicator/strided_sliceStridedSlice emotion_indicator/Shape:output:0.emotion_indicator/strided_slice/stack:output:00emotion_indicator/strided_slice/stack_1:output:00emotion_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!emotion_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
emotion_indicator/Reshape/shapePack(emotion_indicator/strided_slice:output:0*emotion_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
emotion_indicator/ReshapeReshapeemotion_indicator/Sum:output:0(emotion_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
gender_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gender_indicator/ExpandDims
ExpandDimsfeatures_gender(gender_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????p
/gender_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
)gender_indicator/to_sparse_input/NotEqualNotEqual$gender_indicator/ExpandDims:output:08gender_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
(gender_indicator/to_sparse_input/indicesWhere-gender_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
'gender_indicator/to_sparse_input/valuesGatherNd$gender_indicator/ExpandDims:output:00gender_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
,gender_indicator/to_sparse_input/dense_shapeShape$gender_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
4gender_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Agender_indicator_hash_table_lookup_lookuptablefindv2_table_handle0gender_indicator/to_sparse_input/values:output:0Bgender_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????w
,gender_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
gender_indicator/SparseToDenseSparseToDense0gender_indicator/to_sparse_input/indices:index:05gender_indicator/to_sparse_input/dense_shape:output:0=gender_indicator/hash_table_Lookup/LookupTableFindV2:values:05gender_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????c
gender_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
 gender_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    `
gender_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
gender_indicator/one_hotOneHot&gender_indicator/SparseToDense:dense:0'gender_indicator/one_hot/depth:output:0'gender_indicator/one_hot/Const:output:0)gender_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????y
&gender_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
gender_indicator/SumSum!gender_indicator/one_hot:output:0/gender_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????c
gender_indicator/ShapeShapegender_indicator/Sum:output:0*
T0*
_output_shapes
:n
$gender_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&gender_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&gender_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gender_indicator/strided_sliceStridedSlicegender_indicator/Shape:output:0-gender_indicator/strided_slice/stack:output:0/gender_indicator/strided_slice/stack_1:output:0/gender_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 gender_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
gender_indicator/Reshape/shapePack'gender_indicator/strided_slice:output:0)gender_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
gender_indicator/ReshapeReshapegender_indicator/Sum:output:0'gender_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$nationality_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 nationality_indicator/ExpandDims
ExpandDimsfeatures_nationality-nationality_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4nationality_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.nationality_indicator/to_sparse_input/NotEqualNotEqual)nationality_indicator/ExpandDims:output:0=nationality_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-nationality_indicator/to_sparse_input/indicesWhere2nationality_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,nationality_indicator/to_sparse_input/valuesGatherNd)nationality_indicator/ExpandDims:output:05nationality_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1nationality_indicator/to_sparse_input/dense_shapeShape)nationality_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
9nationality_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Fnationality_indicator_hash_table_lookup_lookuptablefindv2_table_handle5nationality_indicator/to_sparse_input/values:output:0Gnationality_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1nationality_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#nationality_indicator/SparseToDenseSparseToDense5nationality_indicator/to_sparse_input/indices:index:0:nationality_indicator/to_sparse_input/dense_shape:output:0Bnationality_indicator/hash_table_Lookup/LookupTableFindV2:values:0:nationality_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#nationality_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%nationality_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#nationality_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
nationality_indicator/one_hotOneHot+nationality_indicator/SparseToDense:dense:0,nationality_indicator/one_hot/depth:output:0,nationality_indicator/one_hot/Const:output:0.nationality_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+nationality_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
nationality_indicator/SumSum&nationality_indicator/one_hot:output:04nationality_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
nationality_indicator/ShapeShape"nationality_indicator/Sum:output:0*
T0*
_output_shapes
:s
)nationality_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+nationality_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+nationality_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#nationality_indicator/strided_sliceStridedSlice$nationality_indicator/Shape:output:02nationality_indicator/strided_slice/stack:output:04nationality_indicator/strided_slice/stack_1:output:04nationality_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%nationality_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#nationality_indicator/Reshape/shapePack,nationality_indicator/strided_slice:output:0.nationality_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
nationality_indicator/ReshapeReshape"nationality_indicator/Sum:output:0,nationality_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????n
#occupation_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
occupation_indicator/ExpandDims
ExpandDimsfeatures_occupation,occupation_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3occupation_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-occupation_indicator/to_sparse_input/NotEqualNotEqual(occupation_indicator/ExpandDims:output:0<occupation_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,occupation_indicator/to_sparse_input/indicesWhere1occupation_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+occupation_indicator/to_sparse_input/valuesGatherNd(occupation_indicator/ExpandDims:output:04occupation_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0occupation_indicator/to_sparse_input/dense_shapeShape(occupation_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
8occupation_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Eoccupation_indicator_hash_table_lookup_lookuptablefindv2_table_handle4occupation_indicator/to_sparse_input/values:output:0Foccupation_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0occupation_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"occupation_indicator/SparseToDenseSparseToDense4occupation_indicator/to_sparse_input/indices:index:09occupation_indicator/to_sparse_input/dense_shape:output:0Aoccupation_indicator/hash_table_Lookup/LookupTableFindV2:values:09occupation_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"occupation_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$occupation_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"occupation_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
occupation_indicator/one_hotOneHot*occupation_indicator/SparseToDense:dense:0+occupation_indicator/one_hot/depth:output:0+occupation_indicator/one_hot/Const:output:0-occupation_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*occupation_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
occupation_indicator/SumSum%occupation_indicator/one_hot:output:03occupation_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
occupation_indicator/ShapeShape!occupation_indicator/Sum:output:0*
T0*
_output_shapes
:r
(occupation_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*occupation_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*occupation_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"occupation_indicator/strided_sliceStridedSlice#occupation_indicator/Shape:output:01occupation_indicator/strided_slice/stack:output:03occupation_indicator/strided_slice/stack_1:output:03occupation_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$occupation_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"occupation_indicator/Reshape/shapePack+occupation_indicator/strided_slice:output:0-occupation_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
occupation_indicator/ReshapeReshape!occupation_indicator/Sum:output:0+occupation_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
topic_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
topic_indicator/ExpandDims
ExpandDimsfeatures_topic'topic_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????o
.topic_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
(topic_indicator/to_sparse_input/NotEqualNotEqual#topic_indicator/ExpandDims:output:07topic_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
'topic_indicator/to_sparse_input/indicesWhere,topic_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
&topic_indicator/to_sparse_input/valuesGatherNd#topic_indicator/ExpandDims:output:0/topic_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
+topic_indicator/to_sparse_input/dense_shapeShape#topic_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3topic_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@topic_indicator_hash_table_lookup_lookuptablefindv2_table_handle/topic_indicator/to_sparse_input/values:output:0Atopic_indicator_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????v
+topic_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
topic_indicator/SparseToDenseSparseToDense/topic_indicator/to_sparse_input/indices:index:04topic_indicator/to_sparse_input/dense_shape:output:0<topic_indicator/hash_table_Lookup/LookupTableFindV2:values:04topic_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????b
topic_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
topic_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    _
topic_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
topic_indicator/one_hotOneHot%topic_indicator/SparseToDense:dense:0&topic_indicator/one_hot/depth:output:0&topic_indicator/one_hot/Const:output:0(topic_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????x
%topic_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
topic_indicator/SumSum topic_indicator/one_hot:output:0.topic_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????a
topic_indicator/ShapeShapetopic_indicator/Sum:output:0*
T0*
_output_shapes
:m
#topic_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%topic_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%topic_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
topic_indicator/strided_sliceStridedSlicetopic_indicator/Shape:output:0,topic_indicator/strided_slice/stack:output:0.topic_indicator/strided_slice/stack_1:output:0.topic_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
topic_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
topic_indicator/Reshape/shapePack&topic_indicator/strided_slice:output:0(topic_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
topic_indicator/ReshapeReshapetopic_indicator/Sum:output:0&topic_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2age_bucketized/Reshape:output:0"emotion_indicator/Reshape:output:0!gender_indicator/Reshape:output:0&nationality_indicator/Reshape:output:0%occupation_indicator/Reshape:output:0 topic_indicator/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp6^emotion_indicator/hash_table_Lookup/LookupTableFindV25^gender_indicator/hash_table_Lookup/LookupTableFindV2:^nationality_indicator/hash_table_Lookup/LookupTableFindV29^occupation_indicator/hash_table_Lookup/LookupTableFindV24^topic_indicator/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2n
5emotion_indicator/hash_table_Lookup/LookupTableFindV25emotion_indicator/hash_table_Lookup/LookupTableFindV22l
4gender_indicator/hash_table_Lookup/LookupTableFindV24gender_indicator/hash_table_Lookup/LookupTableFindV22v
9nationality_indicator/hash_table_Lookup/LookupTableFindV29nationality_indicator/hash_table_Lookup/LookupTableFindV22t
8occupation_indicator/hash_table_Lookup/LookupTableFindV28occupation_indicator/hash_table_Lookup/LookupTableFindV22j
3topic_indicator/hash_table_Lookup/LookupTableFindV23topic_indicator/hash_table_Lookup/LookupTableFindV2:Q M
#
_output_shapes
:?????????
&
_user_specified_namefeatures/age:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/emotion:TP
#
_output_shapes
:?????????
)
_user_specified_namefeatures/gender:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/nationality:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/occupation:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/topic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_3_layer_call_fn_465670

inputs_age	
inputs_emotion
inputs_gender
inputs_nationality
inputs_occupation
inputs_topic
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9:
??

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_ageinputs_emotioninputs_genderinputs_nationalityinputs_occupationinputs_topicunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*#
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_464984o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/age:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/emotion:RN
#
_output_shapes
:?????????
'
_user_specified_nameinputs/gender:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/nationality:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/topic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_dense_14_layer_call_fn_466513

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_464960p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_dense_12_layer_call_fn_466473

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_464926p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_13_layer_call_and_return_conditional_losses_464943

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_dense_features_layer_call_fn_466148
features_age	
features_emotion
features_gender
features_nationality
features_occupation
features_topic
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_agefeatures_emotionfeatures_genderfeatures_nationalityfeatures_occupationfeatures_topicunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2						*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_465259p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
#
_output_shapes
:?????????
&
_user_specified_namefeatures/age:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/emotion:TP
#
_output_shapes
:?????????
)
_user_specified_namefeatures/gender:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/nationality:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/occupation:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/topic:

_output_shapes
: :	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_dense_14_layer_call_and_return_conditional_losses_464960

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_14_layer_call_and_return_conditional_losses_466524

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
-
__inference__destroyer_466598
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_4665932
.table_init197_lookuptableimportv2_table_handle*
&table_init197_lookuptableimportv2_keys,
(table_init197_lookuptableimportv2_values	
identity??!table_init197/LookupTableImportV2?
!table_init197/LookupTableImportV2LookupTableImportV2.table_init197_lookuptableimportv2_table_handle&table_init197_lookuptableimportv2_keys(table_init197_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init197/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2F
!table_init197/LookupTableImportV2!table_init197/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?"?L
saver_filename:0StatefulPartitionedCall_6:0StatefulPartitionedCall_78"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
/
age(
serving_default_age:0	?????????
7
emotion,
serving_default_emotion:0?????????
5
gender+
serving_default_gender:0?????????
?
nationality0
serving_default_nationality:0?????????
=

occupation/
serving_default_occupation:0?????????
3
topic*
serving_default_topic:0?????????>
output_12
StatefulPartitionedCall_5:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
_build_input_shape
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
x__call__
*y&call_and_return_all_conditional_losses
z_default_save_signature"
_tf_keras_sequential
?
_feature_columns

_resources
	variables
trainable_variables
regularization_losses
	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
+iter

,beta_1

-beta_2
	.decay
/learning_ratemhmimjmkml mm%mn&movpvqvrvsvt vu%vv&vw"
	optimizer
 "
trackable_dict_wrapper
X
0
1
2
3
4
 5
%6
&7"
trackable_list_wrapper
X
0
1
2
3
4
 5
%6
&7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
	trainable_variables

regularization_losses
x__call__
z_default_save_signature
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
c
5emotion

6gender
7nationality
8
occupation
	9topic"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
0:.
??2sequential_3/dense_12/kernel
):'?2sequential_3/dense_12/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
0:.
??2sequential_3/dense_13/kernel
):'?2sequential_3/dense_13/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.
??2sequential_3/dense_14/kernel
):'?2sequential_3/dense_14/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
!	variables
"trainable_variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-	?2sequential_3/dense_15/kernel
(:&2sequential_3/dense_15/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
'	variables
(trainable_variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
2
Uemotion_lookup"
_generic_user_object
1
Vgender_lookup"
_generic_user_object
6
Wnationality_lookup"
_generic_user_object
5
Xoccupation_lookup"
_generic_user_object
0
Ytopic_lookup"
_generic_user_object
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
N
	Ztotal
	[count
\	variables
]	keras_api"
_tf_keras_metric
^
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api"
_tf_keras_metric
m
c_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
m
d_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
m
e_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
m
f_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
m
g_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
:  (2total
:  (2count
.
Z0
[1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
5:3
??2#Adam/sequential_3/dense_12/kernel/m
.:,?2!Adam/sequential_3/dense_12/bias/m
5:3
??2#Adam/sequential_3/dense_13/kernel/m
.:,?2!Adam/sequential_3/dense_13/bias/m
5:3
??2#Adam/sequential_3/dense_14/kernel/m
.:,?2!Adam/sequential_3/dense_14/bias/m
4:2	?2#Adam/sequential_3/dense_15/kernel/m
-:+2!Adam/sequential_3/dense_15/bias/m
5:3
??2#Adam/sequential_3/dense_12/kernel/v
.:,?2!Adam/sequential_3/dense_12/bias/v
5:3
??2#Adam/sequential_3/dense_13/kernel/v
.:,?2!Adam/sequential_3/dense_13/bias/v
5:3
??2#Adam/sequential_3/dense_14/kernel/v
.:,?2!Adam/sequential_3/dense_14/bias/v
4:2	?2#Adam/sequential_3/dense_15/kernel/v
-:+2!Adam/sequential_3/dense_15/bias/v
?2?
-__inference_sequential_3_layer_call_fn_465023
-__inference_sequential_3_layer_call_fn_465670
-__inference_sequential_3_layer_call_fn_465716
-__inference_sequential_3_layer_call_fn_465470?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_3_layer_call_and_return_conditional_losses_465902
H__inference_sequential_3_layer_call_and_return_conditional_losses_466088
H__inference_sequential_3_layer_call_and_return_conditional_losses_465520
H__inference_sequential_3_layer_call_and_return_conditional_losses_465570?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_464718ageemotiongendernationality
occupationtopic"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_dense_features_layer_call_fn_466118
/__inference_dense_features_layer_call_fn_466148?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_dense_features_layer_call_and_return_conditional_losses_466306
J__inference_dense_features_layer_call_and_return_conditional_losses_466464?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_12_layer_call_fn_466473?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_12_layer_call_and_return_conditional_losses_466484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_13_layer_call_fn_466493?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_13_layer_call_and_return_conditional_losses_466504?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_14_layer_call_fn_466513?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_14_layer_call_and_return_conditional_losses_466524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_15_layer_call_fn_466533?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_15_layer_call_and_return_conditional_losses_466544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_465624ageemotiongendernationality
occupationtopic"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_466549?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_466557?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_466562?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_466567?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_466575?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_466580?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_466585?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_466593?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_466598?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_466603?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_466611?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_466616?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_466621?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_466629?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_466634?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_147
__inference__creator_466549?

? 
? "? 7
__inference__creator_466567?

? 
? "? 7
__inference__creator_466585?

? 
? "? 7
__inference__creator_466603?

? 
? "? 7
__inference__creator_466621?

? 
? "? 9
__inference__destroyer_466562?

? 
? "? 9
__inference__destroyer_466580?

? 
? "? 9
__inference__destroyer_466598?

? 
? "? 9
__inference__destroyer_466616?

? 
? "? 9
__inference__destroyer_466634?

? 
? "? B
__inference__initializer_466557U???

? 
? "? B
__inference__initializer_466575V???

? 
? "? B
__inference__initializer_466593W???

? 
? "? B
__inference__initializer_466611X???

? 
? "? B
__inference__initializer_466629Y???

? 
? "? ?
!__inference__wrapped_model_464718?U?V?W?X?Y? %&???
???
???
 
age?
age?????????	
(
emotion?
emotion?????????
&
gender?
gender?????????
0
nationality!?
nationality?????????
.

occupation ?

occupation?????????
$
topic?
topic?????????
? "3?0
.
output_1"?
output_1??????????
D__inference_dense_12_layer_call_and_return_conditional_losses_466484^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_12_layer_call_fn_466473Q0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_13_layer_call_and_return_conditional_losses_466504^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_13_layer_call_fn_466493Q0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_14_layer_call_and_return_conditional_losses_466524^ 0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_14_layer_call_fn_466513Q 0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_15_layer_call_and_return_conditional_losses_466544]%&0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_15_layer_call_fn_466533P%&0?-
&?#
!?
inputs??????????
? "???????????
J__inference_dense_features_layer_call_and_return_conditional_losses_466306?U?V?W?X?Y????
???
???
)
age"?
features/age?????????	
1
emotion&?#
features/emotion?????????
/
gender%?"
features/gender?????????
9
nationality*?'
features/nationality?????????
7

occupation)?&
features/occupation?????????
-
topic$?!
features/topic?????????

 
p 
? "&?#
?
0??????????
? ?
J__inference_dense_features_layer_call_and_return_conditional_losses_466464?U?V?W?X?Y????
???
???
)
age"?
features/age?????????	
1
emotion&?#
features/emotion?????????
/
gender%?"
features/gender?????????
9
nationality*?'
features/nationality?????????
7

occupation)?&
features/occupation?????????
-
topic$?!
features/topic?????????

 
p
? "&?#
?
0??????????
? ?
/__inference_dense_features_layer_call_fn_466118?U?V?W?X?Y????
???
???
)
age"?
features/age?????????	
1
emotion&?#
features/emotion?????????
/
gender%?"
features/gender?????????
9
nationality*?'
features/nationality?????????
7

occupation)?&
features/occupation?????????
-
topic$?!
features/topic?????????

 
p 
? "????????????
/__inference_dense_features_layer_call_fn_466148?U?V?W?X?Y????
???
???
)
age"?
features/age?????????	
1
emotion&?#
features/emotion?????????
/
gender%?"
features/gender?????????
9
nationality*?'
features/nationality?????????
7

occupation)?&
features/occupation?????????
-
topic$?!
features/topic?????????

 
p
? "????????????
H__inference_sequential_3_layer_call_and_return_conditional_losses_465520?U?V?W?X?Y? %&???
???
???
 
age?
age?????????	
(
emotion?
emotion?????????
&
gender?
gender?????????
0
nationality!?
nationality?????????
.

occupation ?

occupation?????????
$
topic?
topic?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_465570?U?V?W?X?Y? %&???
???
???
 
age?
age?????????	
(
emotion?
emotion?????????
&
gender?
gender?????????
0
nationality!?
nationality?????????
.

occupation ?

occupation?????????
$
topic?
topic?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_465902?U?V?W?X?Y? %&???
???
???
'
age ?

inputs/age?????????	
/
emotion$?!
inputs/emotion?????????
-
gender#? 
inputs/gender?????????
7
nationality(?%
inputs/nationality?????????
5

occupation'?$
inputs/occupation?????????
+
topic"?
inputs/topic?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_466088?U?V?W?X?Y? %&???
???
???
'
age ?

inputs/age?????????	
/
emotion$?!
inputs/emotion?????????
-
gender#? 
inputs/gender?????????
7
nationality(?%
inputs/nationality?????????
5

occupation'?$
inputs/occupation?????????
+
topic"?
inputs/topic?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_3_layer_call_fn_465023?U?V?W?X?Y? %&???
???
???
 
age?
age?????????	
(
emotion?
emotion?????????
&
gender?
gender?????????
0
nationality!?
nationality?????????
.

occupation ?

occupation?????????
$
topic?
topic?????????
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_465470?U?V?W?X?Y? %&???
???
???
 
age?
age?????????	
(
emotion?
emotion?????????
&
gender?
gender?????????
0
nationality!?
nationality?????????
.

occupation ?

occupation?????????
$
topic?
topic?????????
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_465670?U?V?W?X?Y? %&???
???
???
'
age ?

inputs/age?????????	
/
emotion$?!
inputs/emotion?????????
-
gender#? 
inputs/gender?????????
7
nationality(?%
inputs/nationality?????????
5

occupation'?$
inputs/occupation?????????
+
topic"?
inputs/topic?????????
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_465716?U?V?W?X?Y? %&???
???
???
'
age ?

inputs/age?????????	
/
emotion$?!
inputs/emotion?????????
-
gender#? 
inputs/gender?????????
7
nationality(?%
inputs/nationality?????????
5

occupation'?$
inputs/occupation?????????
+
topic"?
inputs/topic?????????
p

 
? "???????????
$__inference_signature_wrapper_465624?U?V?W?X?Y? %&???
? 
???
 
age?
age?????????	
(
emotion?
emotion?????????
&
gender?
gender?????????
0
nationality!?
nationality?????????
.

occupation ?

occupation?????????
$
topic?
topic?????????"3?0
.
output_1"?
output_1?????????