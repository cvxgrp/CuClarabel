import cupy as cp
import numpy as np
from numba import cuda, njit
from scipy.sparse import csr_matrix

DefaultFloat = np.float64
DefaultInt = np.int64
SOC_NO_EXPANSION_MAX_SIZE = 5  # maximal size of second-order cones in GPU implementation

# Rust-like Option type
Option = type(None)

# Internal constraint RHS limits
_INFINITY_DEFAULT = 1e20
INFINITY = _INFINITY_DEFAULT

def default_infinity():
    global INFINITY
    INFINITY = _INFINITY_DEFAULT

def set_infinity(v):
    global INFINITY
    INFINITY = float(v)

def get_infinity():
    return INFINITY

# List of GPU solvers
gpu_solver_list = ['cudss', 'cudssmixed']

# Version / release info
# include("./version.py")

# API for user cone specifications
# include("./cones/cone_api.py")

# Cone type definitions
# include("./cones/cone_types.py")
# include("./cones/cone_dispatch.py")
# include("./cones/compositecone_type.py")
# include("./gpucones/compositecone_type_gpu.py")

# Core solver components
# include("./abstract_types.py")
# include("./settings.py")
# include("./statuscodes.py")
# include("./chordal/include.py")
# include("./types.py")
# include("./presolver.py")
# include("./variables.py")
# include("./residuals.py")
# include("./problemdata.py")

# Direct LDL linear solve methods
# include("./kktsolvers/direct-ldl/includes.py")

# KKT solvers and solver level kktsystem
# include("./kktsolvers/kktsolver_defaults.py")
# include("./kktsolvers/kktsolver_directldl.py")

# include("./kktsystem.py")
# include("./kktsystem_gpu.py")

# include("./info.py")
# include("./solution.py")

# GPU ldl methods
# include("./kktsolvers/gpu/includes.py")
# include("./kktsolvers/kktsolver_directldl_gpu.py")

# Printing and top level solver
# include("./info_print.py")
# include("./solver.py")

# Conic constraints. Additional
# cone implementations go here
# include("./cones/coneops_defaults.py")
# include("./cones/coneops_zerocone.py")
# include("./cones/coneops_nncone.py")
# include("./cones/coneops_socone.py")
# include("./cones/coneops_psdtrianglecone.py")
# include("./cones/coneops_expcone.py")
# include("./cones/coneops_powcone.py")
# include("./cones/coneops_genpowcone.py")        # Generalized power cone
# include("./cones/coneops_compositecone.py")
# include("./cones/coneops_nonsymmetric_common.py")
# include("./cones/coneops_symmetric_common.py")

# GPU cone implementations
# include("./gpucones/mathutilGPU.py")
# include("./gpucones/coneops_zerocone_gpu.py")
# include("./gpucones/coneops_nncone_gpu.py")
# include("./gpucones/coneops_socone_gpu.py")
# include("./gpucones/coneops_expcone_gpu.py")
# include("./gpucones/coneops_powcone_gpu.py")
# include("./gpucones/coneops_psdtrianglecone_gpu.py")
# include("./gpucones/coneops_compositecone_gpu.py")
# include("./gpucones/coneops_nonsymmetric_common_gpu.py")
# include("./gpucones/augment_socp.py")

# Various algebraic utilities
# include("./utils/mathutils.py")
# include("./utils/csc_assembly.py")

# Data updating
# include("./data_updating.py")

# Optional dependencies
def __init__():
    try:
        import pardiso
        # include("./kktsolvers/direct-ldl/directldl_pardiso.py")
    except ImportError:
        pass

    try:
        import hsl
        # include("./kktsolvers/direct-ldl/directldl_hsl.py")
    except ImportError:
        pass

# JSON I/O
# include("./json.py")

# MathOptInterface for JuMP/Convex.jl
# module MOI  # extensions providing non-standard MOI constraint types
#     include("./MOI_wrapper/MOI_extensions.py")
# end
# module MOIwrapper # our actual MOI interface
#      include("./MOI_wrapper/MOI_wrapper.py")
# end
# const Optimizer{T} = Clarabel.MOIwrapper.Optimizer{T}

# Precompile minimal MOI / native examples
# using SnoopPrecompile
# include("./precompile.py")
# redirect_stdout(devnull) do;
#     SnoopPrecompile.@precompile_all_calls begin
#         __precompile_native()
#         __precompile_moi()
#     end
# end
# __precompile_printfcns()
