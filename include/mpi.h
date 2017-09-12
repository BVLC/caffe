/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*  
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
/* src/include/mpi.h.  Generated from mpi.h.in by configure. */
#ifndef MPI_INCLUDED
#define MPI_INCLUDED

/* user include file for MPI programs */

/* Keep C++ compilers from getting confused */
#if defined(__cplusplus)
extern "C" {
#endif

#define NO_TAGS_WITH_MODIFIERS 1
#undef MPICH_DEFINE_ATTR_TYPE_TYPES
#if defined(__has_attribute)
#  if __has_attribute(pointer_with_type_tag) && \
      __has_attribute(type_tag_for_datatype) && \
      !defined(NO_TAGS_WITH_MODIFIERS) &&\
      !defined(MPICH_NO_ATTR_TYPE_TAGS)
#    define MPICH_DEFINE_ATTR_TYPE_TYPES 1
#    define MPICH_ATTR_POINTER_WITH_TYPE_TAG(buffer_idx, type_idx)  __attribute__((pointer_with_type_tag(MPI,buffer_idx,type_idx)))
#    define MPICH_ATTR_TYPE_TAG(type)                               __attribute__((type_tag_for_datatype(MPI,type)))
#    define MPICH_ATTR_TYPE_TAG_LAYOUT_COMPATIBLE(type)             __attribute__((type_tag_for_datatype(MPI,type,layout_compatible)))
#    define MPICH_ATTR_TYPE_TAG_MUST_BE_NULL()                      __attribute__((type_tag_for_datatype(MPI,void,must_be_null)))
#    include <stddef.h>
#  endif
#endif

#if !defined(MPICH_ATTR_POINTER_WITH_TYPE_TAG)
#  define MPICH_ATTR_POINTER_WITH_TYPE_TAG(buffer_idx, type_idx)
#  define MPICH_ATTR_TYPE_TAG(type)
#  define MPICH_ATTR_TYPE_TAG_LAYOUT_COMPATIBLE(type)
#  define MPICH_ATTR_TYPE_TAG_MUST_BE_NULL()
#endif

#if !defined(INT8_C)
/* stdint.h was not included, see if we can get it */
#  if defined(__cplusplus)
#    if __cplusplus >= 201103
#      include <cstdint>
#    endif
#  endif
#endif

#if !defined(INT8_C)
/* stdint.h was not included, see if we can get it */
#  if defined(__STDC_VERSION__)
#    if __STDC_VERSION__ >= 199901
#      include <stdint.h>
#    endif
#  endif
#endif

#if defined(INT8_C)
/* stdint.h was included, so we can annotate these types */
#  define MPICH_ATTR_TYPE_TAG_STDINT(type) MPICH_ATTR_TYPE_TAG(type)
#else
#  define MPICH_ATTR_TYPE_TAG_STDINT(type)
#endif

#ifdef __STDC_VERSION__ 
#if __STDC_VERSION__ >= 199901
#  define MPICH_ATTR_TYPE_TAG_C99(type) MPICH_ATTR_TYPE_TAG(type)
#else
#  define MPICH_ATTR_TYPE_TAG_C99(type)
#endif
#else 
#  define MPICH_ATTR_TYPE_TAG_C99(type)
#endif

#if defined(__cplusplus)
#  define MPICH_ATTR_TYPE_TAG_CXX(type) MPICH_ATTR_TYPE_TAG(type)
#else
#  define MPICH_ATTR_TYPE_TAG_CXX(type)
#endif


/* Define some null objects */
#define MPI_COMM_NULL      ((MPI_Comm)0x04000000)
#define MPI_OP_NULL        ((MPI_Op)0x18000000)
#define MPI_GROUP_NULL     ((MPI_Group)0x08000000)
#define MPI_DATATYPE_NULL  ((MPI_Datatype)0x0c000000)
#define MPI_REQUEST_NULL   ((MPI_Request)0x2c000000)
#define MPI_ERRHANDLER_NULL ((MPI_Errhandler)0x14000000)
#define MPI_MESSAGE_NULL   ((MPI_Message)0x2c000000)
#define MPI_MESSAGE_NO_PROC ((MPI_Message)0x6c000000)

/* Results of the compare operations. */
#define MPI_IDENT     0
#define MPI_CONGRUENT 1
#define MPI_SIMILAR   2
#define MPI_UNEQUAL   3

typedef int MPI_Datatype;
#define MPI_CHAR           ((MPI_Datatype)0x4c000101)
#define MPI_SIGNED_CHAR    ((MPI_Datatype)0x4c000118)
#define MPI_UNSIGNED_CHAR  ((MPI_Datatype)0x4c000102)
#define MPI_BYTE           ((MPI_Datatype)0x4c00010d)
#define MPI_WCHAR          ((MPI_Datatype)0x4c00040e)
#define MPI_SHORT          ((MPI_Datatype)0x4c000203)
#define MPI_UNSIGNED_SHORT ((MPI_Datatype)0x4c000204)
#define MPI_INT            ((MPI_Datatype)0x4c000405)
#define MPI_UNSIGNED       ((MPI_Datatype)0x4c000406)
#define MPI_LONG           ((MPI_Datatype)0x4c000807)
#define MPI_UNSIGNED_LONG  ((MPI_Datatype)0x4c000808)
#define MPI_FLOAT          ((MPI_Datatype)0x4c00040a)
#define MPI_DOUBLE         ((MPI_Datatype)0x4c00080b)
#define MPI_LONG_DOUBLE    ((MPI_Datatype)0x4c00100c)
#define MPI_LONG_LONG_INT  ((MPI_Datatype)0x4c000809)
#define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)0x4c000819)
#define MPI_LONG_LONG      MPI_LONG_LONG_INT

#ifdef MPICH_DEFINE_ATTR_TYPE_TYPES
static const MPI_Datatype mpich_mpi_char               MPICH_ATTR_TYPE_TAG(char)               = MPI_CHAR;
static const MPI_Datatype mpich_mpi_signed_char        MPICH_ATTR_TYPE_TAG(signed char)        = MPI_SIGNED_CHAR;
static const MPI_Datatype mpich_mpi_unsigned_char      MPICH_ATTR_TYPE_TAG(unsigned char)      = MPI_UNSIGNED_CHAR;
/*static const MPI_Datatype mpich_mpi_byte               MPICH_ATTR_TYPE_TAG(char)               = MPI_BYTE;*/
static const MPI_Datatype mpich_mpi_wchar              MPICH_ATTR_TYPE_TAG(wchar_t)            = MPI_WCHAR;
static const MPI_Datatype mpich_mpi_short              MPICH_ATTR_TYPE_TAG(short)              = MPI_SHORT;
static const MPI_Datatype mpich_mpi_unsigned_short     MPICH_ATTR_TYPE_TAG(unsigned short)     = MPI_UNSIGNED_SHORT;
static const MPI_Datatype mpich_mpi_int                MPICH_ATTR_TYPE_TAG(int)                = MPI_INT;
static const MPI_Datatype mpich_mpi_unsigned           MPICH_ATTR_TYPE_TAG(unsigned)           = MPI_UNSIGNED;
static const MPI_Datatype mpich_mpi_long               MPICH_ATTR_TYPE_TAG(long)               = MPI_LONG;
static const MPI_Datatype mpich_mpi_unsigned_long      MPICH_ATTR_TYPE_TAG(unsigned long)      = MPI_UNSIGNED_LONG;
static const MPI_Datatype mpich_mpi_float              MPICH_ATTR_TYPE_TAG(float)              = MPI_FLOAT;
static const MPI_Datatype mpich_mpi_double             MPICH_ATTR_TYPE_TAG(double)             = MPI_DOUBLE;
#if 0x4c00100c != 0x0c000000
static const MPI_Datatype mpich_mpi_long_double        MPICH_ATTR_TYPE_TAG(long double)        = MPI_LONG_DOUBLE;
#endif
static const MPI_Datatype mpich_mpi_long_long_int      MPICH_ATTR_TYPE_TAG(long long int)      = MPI_LONG_LONG_INT;
static const MPI_Datatype mpich_mpi_unsigned_long_long MPICH_ATTR_TYPE_TAG(unsigned long long) = MPI_UNSIGNED_LONG_LONG;
#endif

#define MPI_PACKED         ((MPI_Datatype)0x4c00010f)
#define MPI_LB             ((MPI_Datatype)0x4c000010)
#define MPI_UB             ((MPI_Datatype)0x4c000011)

/* 
   The layouts for the types MPI_DOUBLE_INT etc are simply
   struct { 
       double var;
       int    loc;
   }
   This is documented in the man pages on the various datatypes.   
 */
#define MPI_FLOAT_INT         ((MPI_Datatype)0x8c000000)
#define MPI_DOUBLE_INT        ((MPI_Datatype)0x8c000001)
#define MPI_LONG_INT          ((MPI_Datatype)0x8c000002)
#define MPI_SHORT_INT         ((MPI_Datatype)0x8c000003)
#define MPI_2INT              ((MPI_Datatype)0x4c000816)
#define MPI_LONG_DOUBLE_INT   ((MPI_Datatype)0x8c000004)

#ifdef MPICH_DEFINE_ATTR_TYPE_TYPES
struct mpich_struct_mpi_float_int       { float f; int i; };
struct mpich_struct_mpi_double_int      { double d; int i; };
struct mpich_struct_mpi_long_int        { long l; int i; };
struct mpich_struct_mpi_short_int       { short s; int i; };
struct mpich_struct_mpi_2int            { int i1; int i2; };
#if 0x8c000004 != 0x0c000000
struct mpich_struct_mpi_long_double_int { long double ld; int i; };
#endif

static const MPI_Datatype mpich_mpi_float_int       MPICH_ATTR_TYPE_TAG_LAYOUT_COMPATIBLE(struct mpich_struct_mpi_float_int)       = MPI_FLOAT_INT;
static const MPI_Datatype mpich_mpi_double_int      MPICH_ATTR_TYPE_TAG_LAYOUT_COMPATIBLE(struct mpich_struct_mpi_double_int)      = MPI_DOUBLE_INT;
static const MPI_Datatype mpich_mpi_long_int        MPICH_ATTR_TYPE_TAG_LAYOUT_COMPATIBLE(struct mpich_struct_mpi_long_int)        = MPI_LONG_INT;
static const MPI_Datatype mpich_mpi_short_int       MPICH_ATTR_TYPE_TAG_LAYOUT_COMPATIBLE(struct mpich_struct_mpi_short_int)       = MPI_SHORT_INT;

/*
 * The MPI_2INT line is commented out because currently Clang 3.3 flags
 * struct {int i1; int i2;} as different from int[2]. But actually these
 * two types are of the same layout. Clang gives a type mismatch warning
 * for a definitely correct code like the following:
 *  int in[2], out[2];
 *  MPI_Reduce(in, out, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
 *
 * So, we disable type checking for MPI_2INT until Clang fixes this bug.
 */

/* static const MPI_Datatype mpich_mpi_2int            MPICH_ATTR_TYPE_TAG_LAYOUT_COMPATIBLE(struct mpich_struct_mpi_2int)            = MPI_2INT
 */

#if 0x8c000004 != 0x0c000000
static const MPI_Datatype mpich_mpi_long_double_int MPICH_ATTR_TYPE_TAG_LAYOUT_COMPATIBLE(struct mpich_struct_mpi_long_double_int) = MPI_LONG_DOUBLE_INT;
#endif
#endif

/* Fortran types */
#define MPI_COMPLEX           ((MPI_Datatype)1275070494)
#define MPI_DOUBLE_COMPLEX    ((MPI_Datatype)1275072546)
#define MPI_LOGICAL           ((MPI_Datatype)1275069469)
#define MPI_REAL              ((MPI_Datatype)1275069468)
#define MPI_DOUBLE_PRECISION  ((MPI_Datatype)1275070495)
#define MPI_INTEGER           ((MPI_Datatype)1275069467)
#define MPI_2INTEGER          ((MPI_Datatype)1275070496)
/* 
 * MPI_2COMPLEX and MPI_2DOUBLE_COMPLEX were defined by accident in 
 * MPI 1.0 and removed in MPI 1.1.  
 *
 * This definition provides backward compatibility.  These definitions
 * will be removed in a subsequent MPICH release
 */
#ifdef MPICH_DEFINE_2COMPLEX
#define MPI_2COMPLEX          ((MPI_Datatype)1275072548)
#define MPI_2DOUBLE_COMPLEX   ((MPI_Datatype)1275076645)
#endif 
#define MPI_2REAL             ((MPI_Datatype)1275070497)
#define MPI_2DOUBLE_PRECISION ((MPI_Datatype)1275072547)
#define MPI_CHARACTER         ((MPI_Datatype)1275068698)

/* Size-specific types (see MPI-2, 10.2.5) */
#define MPI_REAL4             ((MPI_Datatype)0x4c000427)
#define MPI_REAL8             ((MPI_Datatype)0x4c000829)
#define MPI_REAL16            ((MPI_Datatype)0x4c00102b)
#define MPI_COMPLEX8          ((MPI_Datatype)0x4c000828)
#define MPI_COMPLEX16         ((MPI_Datatype)0x4c00102a)
#define MPI_COMPLEX32         ((MPI_Datatype)0x4c00202c)
#define MPI_INTEGER1          ((MPI_Datatype)0x4c00012d)
#define MPI_INTEGER2          ((MPI_Datatype)0x4c00022f)
#define MPI_INTEGER4          ((MPI_Datatype)0x4c000430)
#define MPI_INTEGER8          ((MPI_Datatype)0x4c000831)
#define MPI_INTEGER16         ((MPI_Datatype)MPI_DATATYPE_NULL)

/* C99 fixed-width datatypes */
#define MPI_INT8_T            ((MPI_Datatype)0x4c000137)
#define MPI_INT16_T           ((MPI_Datatype)0x4c000238)
#define MPI_INT32_T           ((MPI_Datatype)0x4c000439)
#define MPI_INT64_T           ((MPI_Datatype)0x4c00083a)
#define MPI_UINT8_T           ((MPI_Datatype)0x4c00013b)
#define MPI_UINT16_T          ((MPI_Datatype)0x4c00023c)
#define MPI_UINT32_T          ((MPI_Datatype)0x4c00043d)
#define MPI_UINT64_T          ((MPI_Datatype)0x4c00083e)

#ifdef MPICH_DEFINE_ATTR_TYPE_TYPES
static const MPI_Datatype mpich_mpi_int8_t   MPICH_ATTR_TYPE_TAG_STDINT(int8_t)   = MPI_INT8_T;
static const MPI_Datatype mpich_mpi_int16_t  MPICH_ATTR_TYPE_TAG_STDINT(int16_t)  = MPI_INT16_T;
static const MPI_Datatype mpich_mpi_int32_t  MPICH_ATTR_TYPE_TAG_STDINT(int32_t)  = MPI_INT32_T;
static const MPI_Datatype mpich_mpi_int64_t  MPICH_ATTR_TYPE_TAG_STDINT(int64_t)  = MPI_INT64_T;
static const MPI_Datatype mpich_mpi_uint8_t  MPICH_ATTR_TYPE_TAG_STDINT(uint8_t)  = MPI_UINT8_T;
static const MPI_Datatype mpich_mpi_uint16_t MPICH_ATTR_TYPE_TAG_STDINT(uint16_t) = MPI_UINT16_T;
static const MPI_Datatype mpich_mpi_uint32_t MPICH_ATTR_TYPE_TAG_STDINT(uint32_t) = MPI_UINT32_T;
static const MPI_Datatype mpich_mpi_uint64_t MPICH_ATTR_TYPE_TAG_STDINT(uint64_t) = MPI_UINT64_T;
#endif

/* other C99 types */
#define MPI_C_BOOL                 ((MPI_Datatype)0x4c00013f)
#define MPI_C_FLOAT_COMPLEX        ((MPI_Datatype)0x4c000840)
#define MPI_C_COMPLEX              MPI_C_FLOAT_COMPLEX
#define MPI_C_DOUBLE_COMPLEX       ((MPI_Datatype)0x4c001041)
#define MPI_C_LONG_DOUBLE_COMPLEX  ((MPI_Datatype)0x4c002042)

#ifdef MPICH_DEFINE_ATTR_TYPE_TYPES
static const MPI_Datatype mpich_mpi_c_bool                MPICH_ATTR_TYPE_TAG_C99(_Bool)           = MPI_C_BOOL;
static const MPI_Datatype mpich_mpi_c_float_complex       MPICH_ATTR_TYPE_TAG_C99(float _Complex)  = MPI_C_FLOAT_COMPLEX;
static const MPI_Datatype mpich_mpi_c_double_complex      MPICH_ATTR_TYPE_TAG_C99(double _Complex) = MPI_C_DOUBLE_COMPLEX;
#if 0x4c002042 != 0x0c000000
static const MPI_Datatype mpich_mpi_c_long_double_complex MPICH_ATTR_TYPE_TAG_C99(long double _Complex) = MPI_C_LONG_DOUBLE_COMPLEX;
#endif
#endif

/* address/offset types */
#define MPI_AINT          ((MPI_Datatype)0x4c000843)
#define MPI_OFFSET        ((MPI_Datatype)0x4c000844)
#define MPI_COUNT         ((MPI_Datatype)0x4c000845)

/* MPI-3 C++ types */
#define MPI_CXX_BOOL                ((MPI_Datatype)0x4c000133)
#define MPI_CXX_FLOAT_COMPLEX       ((MPI_Datatype)0x4c000834)
#define MPI_CXX_DOUBLE_COMPLEX      ((MPI_Datatype)0x4c001035)
#define MPI_CXX_LONG_DOUBLE_COMPLEX ((MPI_Datatype)0x4c002036)

/* typeclasses */
#define MPI_TYPECLASS_REAL 1
#define MPI_TYPECLASS_INTEGER 2
#define MPI_TYPECLASS_COMPLEX 3

/* Communicators */
typedef int MPI_Comm;
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)
#define MPI_COMM_SELF  ((MPI_Comm)0x44000001)

/* Groups */
typedef int MPI_Group;
#define MPI_GROUP_EMPTY ((MPI_Group)0x48000000)

/* RMA and Windows */
typedef int MPI_Win;
#define MPI_WIN_NULL ((MPI_Win)0x20000000)

/* File and IO */
/* This define lets ROMIO know that MPI_File has been defined */
#define MPI_FILE_DEFINED
/* ROMIO uses a pointer for MPI_File objects.  This must be the same definition
   as in src/mpi/romio/include/mpio.h.in  */
typedef struct ADIOI_FileD *MPI_File;
#define MPI_FILE_NULL ((MPI_File)0)

/* Collective operations */
typedef int MPI_Op;

#define MPI_MAX     (MPI_Op)(0x58000001)
#define MPI_MIN     (MPI_Op)(0x58000002)
#define MPI_SUM     (MPI_Op)(0x58000003)
#define MPI_PROD    (MPI_Op)(0x58000004)
#define MPI_LAND    (MPI_Op)(0x58000005)
#define MPI_BAND    (MPI_Op)(0x58000006)
#define MPI_LOR     (MPI_Op)(0x58000007)
#define MPI_BOR     (MPI_Op)(0x58000008)
#define MPI_LXOR    (MPI_Op)(0x58000009)
#define MPI_BXOR    (MPI_Op)(0x5800000a)
#define MPI_MINLOC  (MPI_Op)(0x5800000b)
#define MPI_MAXLOC  (MPI_Op)(0x5800000c)
#define MPI_REPLACE (MPI_Op)(0x5800000d)
#define MPI_NO_OP   (MPI_Op)(0x5800000e)

/* Permanent key values */
/* C Versions (return pointer to value),
   Fortran Versions (return integer value).
   Handled directly by the attribute value routine
   
   DO NOT CHANGE THESE.  The values encode:
   builtin kind (0x1 in bit 30-31)
   Keyval object (0x9 in bits 26-29)
   for communicator (0x1 in bits 22-25)
   
   Fortran versions of the attributes are formed by adding one to
   the C version.
 */
#define MPI_TAG_UB           0x64400001
#define MPI_HOST             0x64400003
#define MPI_IO               0x64400005
#define MPI_WTIME_IS_GLOBAL  0x64400007
#define MPI_UNIVERSE_SIZE    0x64400009
#define MPI_LASTUSEDCODE     0x6440000b
#define MPI_APPNUM           0x6440000d

/* In addition, there are 5 predefined window attributes that are
   defined for every window */
#define MPI_WIN_BASE          0x66000001
#define MPI_WIN_SIZE          0x66000003
#define MPI_WIN_DISP_UNIT     0x66000005
#define MPI_WIN_CREATE_FLAVOR 0x66000007
#define MPI_WIN_MODEL         0x66000009

#ifdef MPICH_DEFINE_ATTR_TYPE_TYPES
static const MPI_Datatype mpich_mpi_datatype_null MPICH_ATTR_TYPE_TAG_MUST_BE_NULL() = MPI_DATATYPE_NULL;
#endif

/* These are only guesses; make sure you change them in mpif.h as well */
#define MPI_MAX_PROCESSOR_NAME 128
#define MPI_MAX_LIBRARY_VERSION_STRING 8192
#define MPI_MAX_ERROR_STRING   512
#define MPI_MAX_PORT_NAME      256
#define MPI_MAX_OBJECT_NAME    128

/* Pre-defined constants */
#define MPI_UNDEFINED      (-32766)
#define MPI_KEYVAL_INVALID 0x24000000

/* MPI-3 window flavors */
typedef enum MPIR_Win_flavor {
    MPI_WIN_FLAVOR_CREATE      = 1,
    MPI_WIN_FLAVOR_ALLOCATE    = 2,
    MPI_WIN_FLAVOR_DYNAMIC     = 3,
    MPI_WIN_FLAVOR_SHARED      = 4
} MPIR_Win_flavor_t;

/* MPI-3 window consistency models */
typedef enum MPIR_Win_model {
    MPI_WIN_SEPARATE   = 1,
    MPI_WIN_UNIFIED    = 2
} MPIR_Win_model_t;

/* Upper bound on the overhead in bsend for each message buffer */
#define MPI_BSEND_OVERHEAD 96

/* Topology types */
typedef enum MPIR_Topo_type { MPI_GRAPH=1, MPI_CART=2, MPI_DIST_GRAPH=3 } MPIR_Topo_type;

#define MPI_BOTTOM      (void *)0
extern int * const MPI_UNWEIGHTED;
extern int * const MPI_WEIGHTS_EMPTY;

#define MPI_PROC_NULL   (-1)
#define MPI_ANY_SOURCE 	(-2)
#define MPI_ROOT        (-3)
#define MPI_ANY_TAG     (-1)

#define MPI_LOCK_EXCLUSIVE  234
#define MPI_LOCK_SHARED     235

/* C functions */
typedef void (MPI_Handler_function) ( MPI_Comm *, int *, ... );
typedef int (MPI_Comm_copy_attr_function)(MPI_Comm, int, void *, void *, 
					  void *, int *);
typedef int (MPI_Comm_delete_attr_function)(MPI_Comm, int, void *, void *);
typedef int (MPI_Type_copy_attr_function)(MPI_Datatype, int, void *, void *, 
					  void *, int *);
typedef int (MPI_Type_delete_attr_function)(MPI_Datatype, int, void *, void *);
typedef int (MPI_Win_copy_attr_function)(MPI_Win, int, void *, void *, void *,
					 int *);
typedef int (MPI_Win_delete_attr_function)(MPI_Win, int, void *, void *);
/* added in MPI-2.2 */
typedef void (MPI_Comm_errhandler_function)(MPI_Comm *, int *, ...);
typedef void (MPI_File_errhandler_function)(MPI_File *, int *, ...);
typedef void (MPI_Win_errhandler_function)(MPI_Win *, int *, ...);
/* names that were added in MPI-2.0 and deprecated in MPI-2.2 */
typedef MPI_Comm_errhandler_function MPI_Comm_errhandler_fn;
typedef MPI_File_errhandler_function MPI_File_errhandler_fn;
typedef MPI_Win_errhandler_function MPI_Win_errhandler_fn;

/* Built in (0x1 in 30-31), errhandler (0x5 in bits 26-29, allkind (0
   in 22-25), index in the low bits */
#define MPI_ERRORS_ARE_FATAL ((MPI_Errhandler)0x54000000)
#define MPI_ERRORS_RETURN    ((MPI_Errhandler)0x54000001)
/* MPIR_ERRORS_THROW_EXCEPTIONS is not part of the MPI standard, it is here to
   facilitate the c++ binding which has MPI::ERRORS_THROW_EXCEPTIONS. 
   Using the MPIR prefix preserved the MPI_ names for objects defined by
   the standard. */
#define MPIR_ERRORS_THROW_EXCEPTIONS ((MPI_Errhandler)0x54000002)
typedef int MPI_Errhandler;

/* Make the C names for the dup function mixed case.
   This is required for systems that use all uppercase names for Fortran 
   externals.  */
/* MPI 1 names */
#define MPI_NULL_COPY_FN   ((MPI_Copy_function *)0)
#define MPI_NULL_DELETE_FN ((MPI_Delete_function *)0)
#define MPI_DUP_FN         MPIR_Dup_fn
/* MPI 2 names */
#define MPI_COMM_NULL_COPY_FN ((MPI_Comm_copy_attr_function*)0)
#define MPI_COMM_NULL_DELETE_FN ((MPI_Comm_delete_attr_function*)0)
#define MPI_COMM_DUP_FN  ((MPI_Comm_copy_attr_function *)MPI_DUP_FN)
#define MPI_WIN_NULL_COPY_FN ((MPI_Win_copy_attr_function*)0)
#define MPI_WIN_NULL_DELETE_FN ((MPI_Win_delete_attr_function*)0)
#define MPI_WIN_DUP_FN   ((MPI_Win_copy_attr_function*)MPI_DUP_FN)
#define MPI_TYPE_NULL_COPY_FN ((MPI_Type_copy_attr_function*)0)
#define MPI_TYPE_NULL_DELETE_FN ((MPI_Type_delete_attr_function*)0)
#define MPI_TYPE_DUP_FN ((MPI_Type_copy_attr_function*)MPI_DUP_FN)

/* MPI request opjects */
typedef int MPI_Request;

/* MPI message objects for Mprobe and related functions */
typedef int MPI_Message;

/* User combination function */
typedef void (MPI_User_function) ( void *, void *, int *, MPI_Datatype * ); 

/* MPI Attribute copy and delete functions */
typedef int (MPI_Copy_function) ( MPI_Comm, int, void *, void *, void *, int * );
typedef int (MPI_Delete_function) ( MPI_Comm, int, void *, void * );

#define MPI_VERSION    3
#define MPI_SUBVERSION 1
#define MPICH_NAME     3
#define MPICH         1
#define MPICH_HAS_C2F  1


/* MPICH_VERSION is the version string. MPICH_NUMVERSION is the
 * numeric version that can be used in numeric comparisons.
 *
 * MPICH_VERSION uses the following format:
 * Version: [MAJ].[MIN].[REV][EXT][EXT_NUMBER]
 * Example: 1.0.7rc1 has
 *          MAJ = 1
 *          MIN = 0
 *          REV = 7
 *          EXT = rc
 *          EXT_NUMBER = 1
 *
 * MPICH_NUMVERSION will convert EXT to a format number:
 *          ALPHA (a) = 0
 *          BETA (b)  = 1
 *          RC (rc)   = 2
 *          PATCH (p) = 3
 * Regular releases are treated as patch 0
 *
 * Numeric version will have 1 digit for MAJ, 2 digits for MIN, 2
 * digits for REV, 1 digit for EXT and 2 digits for EXT_NUMBER. So,
 * 1.0.7rc1 will have the numeric version 10007201.
 */
#define MPICH_VERSION "3.2"
#define MPICH_NUMVERSION 30200300

#define MPICH_RELEASE_TYPE_ALPHA  0
#define MPICH_RELEASE_TYPE_BETA   1
#define MPICH_RELEASE_TYPE_RC     2
#define MPICH_RELEASE_TYPE_PATCH  3

#define MPICH_CALC_VERSION(MAJOR, MINOR, REVISION, TYPE, PATCH) \
    (((MAJOR) * 10000000) + ((MINOR) * 100000) + ((REVISION) * 1000) + ((TYPE) * 100) + (PATCH))

/* for the datatype decoders */
enum MPIR_Combiner_enum {
    MPI_COMBINER_NAMED            = 1,
    MPI_COMBINER_DUP              = 2,
    MPI_COMBINER_CONTIGUOUS       = 3, 
    MPI_COMBINER_VECTOR           = 4,
    MPI_COMBINER_HVECTOR_INTEGER  = 5,
    MPI_COMBINER_HVECTOR          = 6,
    MPI_COMBINER_INDEXED          = 7,
    MPI_COMBINER_HINDEXED_INTEGER = 8, 
    MPI_COMBINER_HINDEXED         = 9, 
    MPI_COMBINER_INDEXED_BLOCK    = 10, 
    MPI_COMBINER_STRUCT_INTEGER   = 11,
    MPI_COMBINER_STRUCT           = 12,
    MPI_COMBINER_SUBARRAY         = 13,
    MPI_COMBINER_DARRAY           = 14,
    MPI_COMBINER_F90_REAL         = 15,
    MPI_COMBINER_F90_COMPLEX      = 16,
    MPI_COMBINER_F90_INTEGER      = 17,
    MPI_COMBINER_RESIZED          = 18,
    MPI_COMBINER_HINDEXED_BLOCK   = 19
};

/* for info */
typedef int MPI_Info;
#define MPI_INFO_NULL         ((MPI_Info)0x1c000000)
#define MPI_INFO_ENV          ((MPI_Info)0x5c000001)
#define MPI_MAX_INFO_KEY       255
#define MPI_MAX_INFO_VAL      1024

/* for subarray and darray constructors */
#define MPI_ORDER_C              56
#define MPI_ORDER_FORTRAN        57
#define MPI_DISTRIBUTE_BLOCK    121
#define MPI_DISTRIBUTE_CYCLIC   122
#define MPI_DISTRIBUTE_NONE     123
#define MPI_DISTRIBUTE_DFLT_DARG -49767

#define MPI_IN_PLACE  (void *) -1

/* asserts for one-sided communication */
#define MPI_MODE_NOCHECK      1024
#define MPI_MODE_NOSTORE      2048
#define MPI_MODE_NOPUT        4096
#define MPI_MODE_NOPRECEDE    8192
#define MPI_MODE_NOSUCCEED   16384 

/* predefined types for MPI_Comm_split_type */
#define MPI_COMM_TYPE_SHARED    1

/* Definitions that are determined by configure. */
typedef long MPI_Aint;
typedef int MPI_Fint;
typedef long long MPI_Count;

#ifdef MPICH_DEFINE_ATTR_TYPE_TYPES
static const MPI_Datatype mpich_mpi_aint   MPICH_ATTR_TYPE_TAG(MPI_Aint)   = MPI_AINT;
#endif

/* FIXME: The following two definition are not defined by MPI and must not be
   included in the mpi.h file, as the MPI namespace is reserved to the MPI 
   standard */
#define MPI_AINT_FMT_DEC_SPEC "%ld"
#define MPI_AINT_FMT_HEX_SPEC "%lx"

/* Let ROMIO know that MPI_Offset is already defined */
#define HAVE_MPI_OFFSET
/* MPI_OFFSET_TYPEDEF is set in configure and is 
      typedef $MPI_OFFSET MPI_Offset;
   where $MPI_OFFSET is the correct C type */
typedef long long MPI_Offset;

#ifdef MPICH_DEFINE_ATTR_TYPE_TYPES
static const MPI_Datatype mpich_mpi_offset MPICH_ATTR_TYPE_TAG(MPI_Offset) = MPI_OFFSET;
#endif

/* The order of these elements must match that in mpif.h, mpi_f08_types.f90,
   and mpi_c_interface_types.f90 */
typedef struct MPI_Status {
    int count_lo;
    int count_hi_and_cancelled;
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
} MPI_Status;

/* types for the MPI_T_ interface */
struct MPIR_T_enum_s;
struct MPIR_T_cvar_handle_s;
struct MPIR_T_pvar_handle_s;
struct MPIR_T_pvar_session_s;

typedef struct MPIR_T_enum_s * MPI_T_enum;
typedef struct MPIR_T_cvar_handle_s * MPI_T_cvar_handle;
typedef struct MPIR_T_pvar_handle_s * MPI_T_pvar_handle;
typedef struct MPIR_T_pvar_session_s * MPI_T_pvar_session;

/* extra const at front would be safer, but is incompatible with MPI_T_ prototypes */
extern struct MPIR_T_pvar_handle_s * const MPI_T_PVAR_ALL_HANDLES;

#define MPI_T_ENUM_NULL         ((MPI_T_enum)NULL)
#define MPI_T_CVAR_HANDLE_NULL  ((MPI_T_cvar_handle)NULL)
#define MPI_T_PVAR_HANDLE_NULL  ((MPI_T_pvar_handle)NULL)
#define MPI_T_PVAR_SESSION_NULL ((MPI_T_pvar_session)NULL)

/* the MPI_T_ interface requires that these VERBOSITY constants occur in this
 * relative order with increasing values */
typedef enum MPIR_T_verbosity_t {
    /* don't name-shift this if/when MPI_T_ is accepted, this is an MPICH-only
     * extension */
    MPIX_T_VERBOSITY_INVALID = 0,

    /* arbitrarily shift values to aid debugging and reduce accidental errors */
    MPI_T_VERBOSITY_USER_BASIC = 221,
    MPI_T_VERBOSITY_USER_DETAIL,
    MPI_T_VERBOSITY_USER_ALL,

    MPI_T_VERBOSITY_TUNER_BASIC,
    MPI_T_VERBOSITY_TUNER_DETAIL,
    MPI_T_VERBOSITY_TUNER_ALL,

    MPI_T_VERBOSITY_MPIDEV_BASIC,
    MPI_T_VERBOSITY_MPIDEV_DETAIL,
    MPI_T_VERBOSITY_MPIDEV_ALL
} MPIR_T_verbosity_t;

typedef enum MPIR_T_bind_t {
    /* don't name-shift this if/when MPI_T_ is accepted, this is an MPICH-only
     * extension */
    MPIX_T_BIND_INVALID = 0,

    /* arbitrarily shift values to aid debugging and reduce accidental errors */
    MPI_T_BIND_NO_OBJECT = 9700,
    MPI_T_BIND_MPI_COMM,
    MPI_T_BIND_MPI_DATATYPE,
    MPI_T_BIND_MPI_ERRHANDLER,
    MPI_T_BIND_MPI_FILE,
    MPI_T_BIND_MPI_GROUP,
    MPI_T_BIND_MPI_OP,
    MPI_T_BIND_MPI_REQUEST,
    MPI_T_BIND_MPI_WIN,
    MPI_T_BIND_MPI_MESSAGE,
    MPI_T_BIND_MPI_INFO
} MPIR_T_bind_t;

typedef enum MPIR_T_scope_t {
    /* don't name-shift this if/when MPI_T_ is accepted, this is an MPICH-only
     * extension */
    MPIX_T_SCOPE_INVALID = 0,

    /* arbitrarily shift values to aid debugging and reduce accidental errors */
    MPI_T_SCOPE_CONSTANT = 60438,
    MPI_T_SCOPE_READONLY,
    MPI_T_SCOPE_LOCAL,
    MPI_T_SCOPE_GROUP,
    MPI_T_SCOPE_GROUP_EQ,
    MPI_T_SCOPE_ALL,
    MPI_T_SCOPE_ALL_EQ
} MPIR_T_scope_t;

typedef enum MPIR_T_pvar_class_t {
    /* don't name-shift this if/when MPI_T_ is accepted, this is an MPICH-only
     * extension */
    MPIX_T_PVAR_CLASS_INVALID = 0,

    /* arbitrarily shift values to aid debugging and reduce accidental errors */
    MPIR_T_PVAR_CLASS_FIRST = 240,
    MPI_T_PVAR_CLASS_STATE = MPIR_T_PVAR_CLASS_FIRST,
    MPI_T_PVAR_CLASS_LEVEL,
    MPI_T_PVAR_CLASS_SIZE,
    MPI_T_PVAR_CLASS_PERCENTAGE,
    MPI_T_PVAR_CLASS_HIGHWATERMARK,
    MPI_T_PVAR_CLASS_LOWWATERMARK,
    MPI_T_PVAR_CLASS_COUNTER,
    MPI_T_PVAR_CLASS_AGGREGATE,
    MPI_T_PVAR_CLASS_TIMER,
    MPI_T_PVAR_CLASS_GENERIC,
    MPIR_T_PVAR_CLASS_LAST,
    MPIR_T_PVAR_CLASS_NUMBER = MPIR_T_PVAR_CLASS_LAST - MPIR_T_PVAR_CLASS_FIRST
} MPIR_T_pvar_class_t;

/* Handle conversion types/functions */

/* Programs that need to convert types used in MPICH should use these */
#define MPI_Comm_c2f(comm) (MPI_Fint)(comm)
#define MPI_Comm_f2c(comm) (MPI_Comm)(comm)
#define MPI_Type_c2f(datatype) (MPI_Fint)(datatype)
#define MPI_Type_f2c(datatype) (MPI_Datatype)(datatype)
#define MPI_Group_c2f(group) (MPI_Fint)(group)
#define MPI_Group_f2c(group) (MPI_Group)(group)
#define MPI_Info_c2f(info) (MPI_Fint)(info)
#define MPI_Info_f2c(info) (MPI_Info)(info)
#define MPI_Request_f2c(request) (MPI_Request)(request)
#define MPI_Request_c2f(request) (MPI_Fint)(request)
#define MPI_Op_c2f(op) (MPI_Fint)(op)
#define MPI_Op_f2c(op) (MPI_Op)(op)
#define MPI_Errhandler_c2f(errhandler) (MPI_Fint)(errhandler)
#define MPI_Errhandler_f2c(errhandler) (MPI_Errhandler)(errhandler)
#define MPI_Win_c2f(win)   (MPI_Fint)(win)
#define MPI_Win_f2c(win)   (MPI_Win)(win)
#define MPI_Message_c2f(msg) ((MPI_Fint)(msg))
#define MPI_Message_f2c(msg) ((MPI_Message)(msg))

/* PMPI versions of the handle transfer functions.  See section 4.17 */
#define PMPI_Comm_c2f(comm) (MPI_Fint)(comm)
#define PMPI_Comm_f2c(comm) (MPI_Comm)(comm)
#define PMPI_Type_c2f(datatype) (MPI_Fint)(datatype)
#define PMPI_Type_f2c(datatype) (MPI_Datatype)(datatype)
#define PMPI_Group_c2f(group) (MPI_Fint)(group)
#define PMPI_Group_f2c(group) (MPI_Group)(group)
#define PMPI_Info_c2f(info) (MPI_Fint)(info)
#define PMPI_Info_f2c(info) (MPI_Info)(info)
#define PMPI_Request_f2c(request) (MPI_Request)(request)
#define PMPI_Request_c2f(request) (MPI_Fint)(request)
#define PMPI_Op_c2f(op) (MPI_Fint)(op)
#define PMPI_Op_f2c(op) (MPI_Op)(op)
#define PMPI_Errhandler_c2f(errhandler) (MPI_Fint)(errhandler)
#define PMPI_Errhandler_f2c(errhandler) (MPI_Errhandler)(errhandler)
#define PMPI_Win_c2f(win)   (MPI_Fint)(win)
#define PMPI_Win_f2c(win)   (MPI_Win)(win)
#define PMPI_Message_c2f(msg) ((MPI_Fint)(msg))
#define PMPI_Message_f2c(msg) ((MPI_Message)(msg))

#define MPI_STATUS_IGNORE (MPI_Status *)1
#define MPI_STATUSES_IGNORE (MPI_Status *)1
#define MPI_ERRCODES_IGNORE (int *)0

/* See 4.12.5 for MPI_F_STATUS(ES)_IGNORE */
#define MPIU_DLL_SPEC
extern MPIU_DLL_SPEC MPI_Fint * MPI_F_STATUS_IGNORE;
extern MPIU_DLL_SPEC MPI_Fint * MPI_F_STATUSES_IGNORE;
/* The annotation MPIU_DLL_SPEC to the extern statements is used 
   as a hook for systems that require C extensions to correctly construct
   DLLs, and is defined as an empty string otherwise
 */

/* The MPI standard requires that the ARGV_NULL values be the same as
   NULL (see 5.3.2) */
#define MPI_ARGV_NULL (char **)0
#define MPI_ARGVS_NULL (char ***)0

/* C type for MPI_STATUS in F08.
   The field order should match that in mpi_f08_types.f90, and mpi_c_interface_types.f90.
 */
typedef struct {
    MPI_Fint count_lo;
    MPI_Fint count_hi_and_cancelled;
    MPI_Fint MPI_SOURCE;
    MPI_Fint MPI_TAG;
    MPI_Fint MPI_ERROR;
} MPI_F08_Status;

extern MPI_F08_Status MPIR_F08_MPI_STATUS_IGNORE_OBJ;
extern MPI_F08_Status MPIR_F08_MPI_STATUSES_IGNORE_OBJ[1];
extern int MPIR_F08_MPI_IN_PLACE;
extern int MPIR_F08_MPI_BOTTOM;

/* Pointers to above objects */
extern MPI_F08_Status *MPI_F08_STATUS_IGNORE;
extern MPI_F08_Status *MPI_F08_STATUSES_IGNORE;

/* For supported thread levels */
#define MPI_THREAD_SINGLE 0
#define MPI_THREAD_FUNNELED 1
#define MPI_THREAD_SERIALIZED 2
#define MPI_THREAD_MULTIPLE 3

/* Typedefs for generalized requests */
typedef int (MPI_Grequest_cancel_function)(void *, int); 
typedef int (MPI_Grequest_free_function)(void *); 
typedef int (MPI_Grequest_query_function)(void *, MPI_Status *); 
typedef int (MPIX_Grequest_poll_function)(void *, MPI_Status *);
typedef int (MPIX_Grequest_wait_function)(int, void **, double, MPI_Status *);

/* MPI's error classes */
#define MPI_SUCCESS          0      /* Successful return code */
/* Communication argument parameters */
#define MPI_ERR_BUFFER       1      /* Invalid buffer pointer */
#define MPI_ERR_COUNT        2      /* Invalid count argument */
#define MPI_ERR_TYPE         3      /* Invalid datatype argument */
#define MPI_ERR_TAG          4      /* Invalid tag argument */
#define MPI_ERR_COMM         5      /* Invalid communicator */
#define MPI_ERR_RANK         6      /* Invalid rank */
#define MPI_ERR_ROOT         7      /* Invalid root */
#define MPI_ERR_TRUNCATE    14      /* Message truncated on receive */

/* MPI Objects (other than COMM) */
#define MPI_ERR_GROUP        8      /* Invalid group */
#define MPI_ERR_OP           9      /* Invalid operation */
#define MPI_ERR_REQUEST     19      /* Invalid mpi_request handle */

/* Special topology argument parameters */
#define MPI_ERR_TOPOLOGY    10      /* Invalid topology */
#define MPI_ERR_DIMS        11      /* Invalid dimension argument */

/* All other arguments.  This is a class with many kinds */
#define MPI_ERR_ARG         12      /* Invalid argument */

/* Other errors that are not simply an invalid argument */
#define MPI_ERR_OTHER       15      /* Other error; use Error_string */

#define MPI_ERR_UNKNOWN     13      /* Unknown error */
#define MPI_ERR_INTERN      16      /* Internal error code    */

/* Multiple completion has three special error classes */
#define MPI_ERR_IN_STATUS           17      /* Look in status for error value */
#define MPI_ERR_PENDING             18      /* Pending request */

/* New MPI-2 Error classes */
#define MPI_ERR_ACCESS      20      /* */
#define MPI_ERR_AMODE       21      /* */
#define MPI_ERR_BAD_FILE    22      /* */
#define MPI_ERR_CONVERSION  23      /* */
#define MPI_ERR_DUP_DATAREP 24      /* */
#define MPI_ERR_FILE_EXISTS 25      /* */
#define MPI_ERR_FILE_IN_USE 26      /* */
#define MPI_ERR_FILE        27      /* */
#define MPI_ERR_IO          32      /* */
#define MPI_ERR_NO_SPACE    36      /* */
#define MPI_ERR_NO_SUCH_FILE 37     /* */
#define MPI_ERR_READ_ONLY   40      /* */
#define MPI_ERR_UNSUPPORTED_DATAREP   43  /* */

/* MPI_ERR_INFO is NOT defined in the MPI-2 standard.  I believe that
   this is an oversight */
#define MPI_ERR_INFO        28      /* */
#define MPI_ERR_INFO_KEY    29      /* */
#define MPI_ERR_INFO_VALUE  30      /* */
#define MPI_ERR_INFO_NOKEY  31      /* */

#define MPI_ERR_NAME        33      /* */
#define MPI_ERR_NO_MEM      34      /* Alloc_mem could not allocate memory */
#define MPI_ERR_NOT_SAME    35      /* */
#define MPI_ERR_PORT        38      /* */
#define MPI_ERR_QUOTA       39      /* */
#define MPI_ERR_SERVICE     41      /* */
#define MPI_ERR_SPAWN       42      /* */
#define MPI_ERR_UNSUPPORTED_OPERATION 44 /* */
#define MPI_ERR_WIN         45      /* */

#define MPI_ERR_BASE        46      /* */
#define MPI_ERR_LOCKTYPE    47      /* */
#define MPI_ERR_KEYVAL      48      /* Erroneous attribute key */
#define MPI_ERR_RMA_CONFLICT 49     /* */
#define MPI_ERR_RMA_SYNC    50      /* */ 
#define MPI_ERR_SIZE        51      /* */
#define MPI_ERR_DISP        52      /* */
#define MPI_ERR_ASSERT      53      /* */

#define MPI_ERR_RMA_RANGE  55       /* */
#define MPI_ERR_RMA_ATTACH 56       /* */
#define MPI_ERR_RMA_SHARED 57       /* */
#define MPI_ERR_RMA_FLAVOR 58       /* */

/* Return codes for functions in the MPI Tool Information Interface */
#define MPI_T_ERR_MEMORY            59  /* Out of memory */
#define MPI_T_ERR_NOT_INITIALIZED   60  /* Interface not initialized */
#define MPI_T_ERR_CANNOT_INIT       61  /* Interface not in the state to
                                           be initialized */
#define MPI_T_ERR_INVALID_INDEX     62  /* The index is invalid or
                                           has been deleted  */
#define MPI_T_ERR_INVALID_ITEM      63  /* Item index queried is out of range */
#define MPI_T_ERR_INVALID_HANDLE    64  /* The handle is invalid */
#define MPI_T_ERR_OUT_OF_HANDLES    65  /* No more handles available */
#define MPI_T_ERR_OUT_OF_SESSIONS   66  /* No more sessions available */
#define MPI_T_ERR_INVALID_SESSION   67  /* Session argument is not valid */
#define MPI_T_ERR_CVAR_SET_NOT_NOW  68  /* Cvar can't be set at this moment */
#define MPI_T_ERR_CVAR_SET_NEVER    69  /* Cvar can't be set until
                                           end of execution */
#define MPI_T_ERR_PVAR_NO_STARTSTOP 70  /* Pvar can't be started or stopped */
#define MPI_T_ERR_PVAR_NO_WRITE     71  /* Pvar can't be written or reset */
#define MPI_T_ERR_PVAR_NO_ATOMIC    72  /* Pvar can't be R/W atomically */
#define MPI_T_ERR_INVALID_NAME      73  /* Name doesn't match */
#define MPI_T_ERR_INVALID           74  /* Generic error code for MPI_T added in MPI-3.1 */


#define MPI_ERR_LASTCODE    0x3fffffff  /* Last valid error code for a 
					   predefined error class */
/* WARNING: this is also defined in mpishared.h.  Update both locations */
#define MPICH_ERR_LAST_CLASS 74     /* It is also helpful to know the
				       last valid class */

#define MPICH_ERR_FIRST_MPIX 100 /* Define a gap here because sock is
                                  * already using some of the values in this
                                  * range. All MPIX error codes will be
                                  * above this value to be ABI complaint. */

#define MPIX_ERR_PROC_FAILED          MPICH_ERR_FIRST_MPIX+1 /* Process failure */
#define MPIX_ERR_PROC_FAILED_PENDING  MPICH_ERR_FIRST_MPIX+2 /* A failure has caused this request
                                                              * to be pending */
#define MPIX_ERR_REVOKED              MPICH_ERR_FIRST_MPIX+3 /* The communciation object has been revoked */

#define MPICH_ERR_LAST_MPIX           MPICH_ERR_FIRST_MPIX+3


/* End of MPI's error classes */

/* Function type defs */
typedef int (MPI_Datarep_conversion_function)(void *, MPI_Datatype, int, 
             void *, MPI_Offset, void *);
typedef int (MPI_Datarep_extent_function)(MPI_Datatype datatype, MPI_Aint *,
                      void *);
#define MPI_CONVERSION_FN_NULL ((MPI_Datarep_conversion_function *)0)

/* 
   For systems that may need to add additional definitions to support
   different declaration styles and options (e.g., different calling 
   conventions or DLL import/export controls).  
*/
/* --Insert Additional Definitions Here-- */

/*
 * Normally, we provide prototypes for all MPI routines.  In a few weird
 * cases, we need to suppress the prototypes.
 */
#ifndef MPICH_SUPPRESS_PROTOTYPES
/* We require that the C compiler support prototypes */
/* Begin Prototypes */
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Buffer_attach(void *buffer, int size);
int MPI_Buffer_detach(void *buffer_addr, int *size);
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Wait(MPI_Request *request, MPI_Status *status);
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
int MPI_Request_free(MPI_Request *request);
int MPI_Waitany(int count, MPI_Request array_of_requests[], int *indx, MPI_Status *status);
int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx, int *flag,
                MPI_Status *status);
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);
int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag,
                MPI_Status array_of_statuses[]);
int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount,
                 int array_of_indices[], MPI_Status array_of_statuses[]);
int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount,
                 int array_of_indices[], MPI_Status array_of_statuses[]);
int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Cancel(MPI_Request *request);
int MPI_Test_cancelled(const MPI_Status *status, int *flag);
int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                  MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                  MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Start(MPI_Request *request);
int MPI_Startall(int count, MPI_Request array_of_requests[]);
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
                 int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag, MPI_Comm comm, MPI_Status *status)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(6,8);
int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest,
                         int sendtag, int source, int recvtag, MPI_Comm comm,
                         MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype,
                    MPI_Datatype *newtype);
int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype,
                     MPI_Datatype *newtype);
int MPI_Type_indexed(int count, const int *array_of_blocklengths,
                     const int *array_of_displacements, MPI_Datatype oldtype,
                     MPI_Datatype *newtype);
int MPI_Type_hindexed(int count, const int *array_of_blocklengths,
                      const MPI_Aint *array_of_displacements, MPI_Datatype oldtype,
                      MPI_Datatype *newtype);
int MPI_Type_struct(int count, const int *array_of_blocklengths,
                    const MPI_Aint *array_of_displacements,
                    const MPI_Datatype *array_of_types, MPI_Datatype *newtype);
int MPI_Address(const void *location, MPI_Aint *address);
int MPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
int MPI_Type_size(MPI_Datatype datatype, int *size);
int MPI_Type_lb(MPI_Datatype datatype, MPI_Aint *displacement);
int MPI_Type_ub(MPI_Datatype datatype, MPI_Aint *displacement);
int MPI_Type_commit(MPI_Datatype *datatype);
int MPI_Type_free(MPI_Datatype *datatype);
int MPI_Get_elements(const MPI_Status *status, MPI_Datatype datatype, int *count);
int MPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf,
             int outsize, int *position, MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Unpack(const void *inbuf, int insize, int *position, void *outbuf, int outcount,
               MPI_Datatype datatype, MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size);
int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
               int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
               MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root,
                MPI_Comm comm)
                MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
                MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,7);
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls,
                  MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
                  const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,8);
int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[],
                  const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                  const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm);
int MPI_Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, MPI_Comm comm)
               MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm)
               MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op);
int MPI_Op_free(MPI_Op *op);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[],
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
                       MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
             MPI_Comm comm)
             MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Group_size(MPI_Group group, int *size);
int MPI_Group_rank(MPI_Group group, int *rank);
int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[], MPI_Group group2,
                              int ranks2[]);
int MPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result);
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);
int MPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);
int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
int MPI_Group_free(MPI_Group *group);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result);
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm);
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
int MPI_Comm_free(MPI_Comm *comm);
int MPI_Comm_test_inter(MPI_Comm comm, int *flag);
int MPI_Comm_remote_size(MPI_Comm comm, int *size);
int MPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm,
                         int remote_leader, int tag, MPI_Comm *newintercomm);
int MPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm);
int MPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                      int *keyval, void *extra_state);
int MPI_Keyval_free(int *keyval);
int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val);
int MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
int MPI_Attr_delete(MPI_Comm comm, int keyval);
int MPI_Topo_test(MPI_Comm comm, int *status);
int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[],
                    int reorder, MPI_Comm *comm_cart);
int MPI_Dims_create(int nnodes, int ndims, int dims[]);
int MPI_Graph_create(MPI_Comm comm_old, int nnodes, const int indx[], const int edges[],
                     int reorder, MPI_Comm *comm_graph);
int MPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges);
int MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int indx[], int edges[]);
int MPI_Cartdim_get(MPI_Comm comm, int *ndims);
int MPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int MPI_Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
int MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[]);
int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest);
int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm);
int MPI_Cart_map(MPI_Comm comm, int ndims, const int dims[], const int periods[], int *newrank);
int MPI_Graph_map(MPI_Comm comm, int nnodes, const int indx[], const int edges[], int *newrank);
int MPI_Get_processor_name(char *name, int *resultlen);
int MPI_Get_version(int *version, int *subversion);
int MPI_Get_library_version(char *version, int *resultlen);
int MPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler);
int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler);
int MPI_Errhandler_free(MPI_Errhandler *errhandler);
int MPI_Error_string(int errorcode, char *string, int *resultlen);
int MPI_Error_class(int errorcode, int *errorclass);
double MPI_Wtime(void);
double MPI_Wtick(void);
int MPI_Init(int *argc, char ***argv);
int MPI_Finalize(void);
int MPI_Initialized(int *flag);
int MPI_Abort(MPI_Comm comm, int errorcode);

/* Note that we may need to define a @PCONTROL_LIST@ depending on whether
   stdargs are supported */
int MPI_Pcontrol(const int level, ...);
int MPI_DUP_FN(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in,
               void *attribute_val_out, int *flag);

/* Process Creation and Management */
int MPI_Close_port(const char *port_name);
int MPI_Comm_accept(const char *port_name, MPI_Info info, int root, MPI_Comm comm,
                    MPI_Comm *newcomm);
int MPI_Comm_connect(const char *port_name, MPI_Info info, int root, MPI_Comm comm,
                     MPI_Comm *newcomm);
int MPI_Comm_disconnect(MPI_Comm *comm);
int MPI_Comm_get_parent(MPI_Comm *parent);
int MPI_Comm_join(int fd, MPI_Comm *intercomm);
int MPI_Comm_spawn(const char *command, char *argv[], int maxprocs, MPI_Info info, int root,
                   MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]);
int MPI_Comm_spawn_multiple(int count, char *array_of_commands[], char **array_of_argv[],
                            const int array_of_maxprocs[], const MPI_Info array_of_info[],
                            int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]);
int MPI_Lookup_name(const char *service_name, MPI_Info info, char *port_name);
int MPI_Open_port(MPI_Info info, char *port_name);
int MPI_Publish_name(const char *service_name, MPI_Info info, const char *port_name);
int MPI_Unpublish_name(const char *service_name, MPI_Info info, const char *port_name);
int MPI_Comm_set_info(MPI_Comm comm, MPI_Info info);
int MPI_Comm_get_info(MPI_Comm comm, MPI_Info *info);

/* One-Sided Communications */
int MPI_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                   int target_rank, MPI_Aint target_disp, int target_count,
                   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count,
            MPI_Datatype target_datatype, MPI_Win win) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count,
            MPI_Datatype target_datatype, MPI_Win win) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Win_complete(MPI_Win win);
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
                   MPI_Win *win);
int MPI_Win_fence(int assert, MPI_Win win);
int MPI_Win_free(MPI_Win *win);
int MPI_Win_get_group(MPI_Win win, MPI_Group *group);
int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win);
int MPI_Win_post(MPI_Group group, int assert, MPI_Win win);
int MPI_Win_start(MPI_Group group, int assert, MPI_Win win);
int MPI_Win_test(MPI_Win win, int *flag);
int MPI_Win_unlock(int rank, MPI_Win win);
int MPI_Win_wait(MPI_Win win);

/* MPI-3 One-Sided Communication Routines */
int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr,
                     MPI_Win *win);
int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
                            void *baseptr, MPI_Win *win);
int MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr);
int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win);
int MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size);
int MPI_Win_detach(MPI_Win win, const void *base);
int MPI_Win_get_info(MPI_Win win, MPI_Info *info_used);
int MPI_Win_set_info(MPI_Win win, MPI_Info info);
int MPI_Get_accumulate(const void *origin_addr, int origin_count,
                        MPI_Datatype origin_datatype, void *result_addr, int result_count,
                        MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                        int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
                        MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                        MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                      MPI_Datatype datatype, int target_rank, MPI_Aint target_disp,
                      MPI_Op op, MPI_Win win)
                      MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                          void *result_addr, MPI_Datatype datatype, int target_rank,
                          MPI_Aint target_disp, MPI_Win win)
                          MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                          MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
                          MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,4);
int MPI_Rput(const void *origin_addr, int origin_count,
              MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype, MPI_Win win,
              MPI_Request *request)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Rget(void *origin_addr, int origin_count,
              MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype, MPI_Win win,
              MPI_Request *request)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Raccumulate(const void *origin_addr, int origin_count,
                     MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
                     int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
                     MPI_Request *request)
                     MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Rget_accumulate(const void *origin_addr, int origin_count,
                         MPI_Datatype origin_datatype, void *result_addr, int result_count,
                         MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                         int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
                         MPI_Request *request)
                         MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                         MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Win_lock_all(int assert, MPI_Win win);
int MPI_Win_unlock_all(MPI_Win win);
int MPI_Win_flush(int rank, MPI_Win win);
int MPI_Win_flush_all(MPI_Win win);
int MPI_Win_flush_local(int rank, MPI_Win win);
int MPI_Win_flush_local_all(MPI_Win win);
int MPI_Win_sync(MPI_Win win);
 
/* External Interfaces */
int MPI_Add_error_class(int *errorclass);
int MPI_Add_error_code(int errorclass, int *errorcode);
int MPI_Add_error_string(int errorcode, const char *string);
int MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode);
int MPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn,
                           MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval,
                           void *extra_state);
int MPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval);
int MPI_Comm_free_keyval(int *comm_keyval);
int MPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag);
int MPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen);
int MPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val);
int MPI_Comm_set_name(MPI_Comm comm, const char *comm_name);
int MPI_File_call_errhandler(MPI_File fh, int errorcode);
int MPI_Grequest_complete(MPI_Request request);
int MPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn,
                       MPI_Grequest_cancel_function *cancel_fn, void *extra_state,
                       MPI_Request *request);
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int MPI_Is_thread_main(int *flag);
int MPI_Query_thread(int *provided);
int MPI_Status_set_cancelled(MPI_Status *status, int flag);
int MPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count);
int MPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn,
                           MPI_Type_delete_attr_function *type_delete_attr_fn,
                           int *type_keyval, void *extra_state);
int MPI_Type_delete_attr(MPI_Datatype datatype, int type_keyval);
int MPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_free_keyval(int *type_keyval);
int MPI_Type_get_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val, int *flag);
int MPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses,
                          int max_datatypes, int array_of_integers[],
                          MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[]);
int MPI_Type_get_envelope(MPI_Datatype datatype, int *num_integers, int *num_addresses,
                          int *num_datatypes, int *combiner);
int MPI_Type_get_name(MPI_Datatype datatype, char *type_name, int *resultlen);
int MPI_Type_set_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val);
int MPI_Type_set_name(MPI_Datatype datatype, const char *type_name);
int MPI_Type_match_size(int typeclass, int size, MPI_Datatype *datatype);
int MPI_Win_call_errhandler(MPI_Win win, int errorcode);
int MPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn,
                          MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval,
                          void *extra_state);
int MPI_Win_delete_attr(MPI_Win win, int win_keyval);
int MPI_Win_free_keyval(int *win_keyval);
int MPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag);
int MPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen);
int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val);
int MPI_Win_set_name(MPI_Win win, const char *win_name);

int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr);
int MPI_Comm_create_errhandler(MPI_Comm_errhandler_function *comm_errhandler_fn,
                               MPI_Errhandler *errhandler);
int MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler);
int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler);
int MPI_File_create_errhandler(MPI_File_errhandler_function *file_errhandler_fn,
                               MPI_Errhandler *errhandler);
int MPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler);
int MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler);
int MPI_Finalized(int *flag);
int MPI_Free_mem(void *base);
int MPI_Get_address(const void *location, MPI_Aint *address);
int MPI_Info_create(MPI_Info *info);
int MPI_Info_delete(MPI_Info info, const char *key);
int MPI_Info_dup(MPI_Info info, MPI_Info *newinfo);
int MPI_Info_free(MPI_Info *info);
int MPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag);
int MPI_Info_get_nkeys(MPI_Info info, int *nkeys);
int MPI_Info_get_nthkey(MPI_Info info, int n, char *key);
int MPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag);
int MPI_Info_set(MPI_Info info, const char *key, const char *value);
int MPI_Pack_external(const char datarep[], const void *inbuf, int incount,
                      MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position)
                      MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Pack_external_size(const char datarep[], int incount, MPI_Datatype datatype,
                           MPI_Aint *size);
int MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status);
int MPI_Status_c2f(const MPI_Status *c_status, MPI_Fint *f_status);
int MPI_Status_f2c(const MPI_Fint *f_status, MPI_Status *c_status);
int MPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[],
                           const int array_of_distribs[], const int array_of_dargs[],
                           const int array_of_psizes[], int order, MPI_Datatype oldtype,
                           MPI_Datatype *newtype);
int MPI_Type_create_hindexed(int count, const int array_of_blocklengths[],
                             const MPI_Aint array_of_displacements[], MPI_Datatype oldtype,
                             MPI_Datatype *newtype);
int MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype,
                            MPI_Datatype *newtype);
int MPI_Type_create_indexed_block(int count, int blocklength, const int array_of_displacements[],
                                  MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_create_hindexed_block(int count, int blocklength,
                                   const MPI_Aint array_of_displacements[],
                                   MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
                            MPI_Datatype *newtype);
int MPI_Type_create_struct(int count, const int array_of_blocklengths[],
                           const MPI_Aint array_of_displacements[],
                           const MPI_Datatype array_of_types[], MPI_Datatype *newtype);
int MPI_Type_create_subarray(int ndims, const int array_of_sizes[],
                             const int array_of_subsizes[], const int array_of_starts[],
                             int order, MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent);
int MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent);
int MPI_Unpack_external(const char datarep[], const void *inbuf, MPI_Aint insize,
                        MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype)
                        MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,7);
int MPI_Win_create_errhandler(MPI_Win_errhandler_function *win_errhandler_fn,
                              MPI_Errhandler *errhandler);
int MPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler);
int MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler);

/* Fortran 90-related functions.  These routines are available only if
   Fortran 90 support is enabled 
*/
int MPI_Type_create_f90_integer(int range, MPI_Datatype *newtype);
int MPI_Type_create_f90_real(int precision, int range, MPI_Datatype *newtype);
int MPI_Type_create_f90_complex(int precision, int range, MPI_Datatype *newtype);

int MPI_Reduce_local(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype,
                     MPI_Op op)
                     MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Op_commutative(MPI_Op op, int *commute);
int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[],
                                   const int sourceweights[], int outdegree,
                                   const int destinations[], const int destweights[],
                                   MPI_Info info, int reorder, MPI_Comm *comm_dist_graph);
int MPI_Dist_graph_create(MPI_Comm comm_old, int n, const int sources[], const int degrees[],
                          const int destinations[], const int weights[], MPI_Info info,
                          int reorder, MPI_Comm *comm_dist_graph);
int MPI_Dist_graph_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree, int *weighted);
int MPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[],
                             int maxoutdegree, int destinations[], int destweights[]);

/* Matched probe functionality */
int MPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message,
                MPI_Status *status);
int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message,
               MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status);
int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message,
              MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);

/* Nonblocking collectives */
int MPI_Comm_idup(MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *request);
int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request);
int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm,
               MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int MPI_Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
                MPI_Request *request)
                MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root,
                 MPI_Comm comm, MPI_Request *request)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int MPI_Iscatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
                 MPI_Request *request)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Iscatterv(const void *sendbuf, const int sendcounts[], const int displs[],
                  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm, MPI_Request *request)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,7);
int MPI_Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                    const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                    MPI_Comm comm, MPI_Request *request)
                    MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int MPI_Ialltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Ialltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                   MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                   const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                   MPI_Request *request)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,8);
int MPI_Ialltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[],
                   const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                   const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm,
                   MPI_Request *request);
int MPI_Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, int root, MPI_Comm comm, MPI_Request *request)
                MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                   MPI_Op op, MPI_Comm comm, MPI_Request *request)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Ireduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[],
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)
                        MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                              MPI_Request *request)
                              MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                              MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Iscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
              MPI_Comm comm, MPI_Request *request)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_Iexscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, MPI_Comm comm, MPI_Request *request)
                MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);

/* Neighborhood collectives */
int MPI_Ineighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            void *recvbuf, int recvcount, MPI_Datatype recvtype,
                            MPI_Comm comm, MPI_Request *request)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Ineighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, const int recvcounts[], const int displs[],
                             MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int MPI_Ineighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm,
                           MPI_Request *request)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Ineighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                            MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                            MPI_Request *request)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,8);
int MPI_Ineighbor_alltoallw(const void *sendbuf, const int sendcounts[],
                            const MPI_Aint sdispls[], const MPI_Datatype sendtypes[],
                            void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[],
                            const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request);
int MPI_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Neighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            void *recvbuf, const int recvcounts[], const int displs[],
                            MPI_Datatype recvtype, MPI_Comm comm)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int MPI_Neighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                          void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                          MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                          MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                           MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                           const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,8);
int MPI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[],
                           const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                           const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm);

/* Shared memory */
int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm);

/* MPI-3 "large count" routines */
int MPI_Get_elements_x(const MPI_Status *status, MPI_Datatype datatype, MPI_Count *count);
int MPI_Status_set_elements_x(MPI_Status *status, MPI_Datatype datatype, MPI_Count count);
int MPI_Type_get_extent_x(MPI_Datatype datatype, MPI_Count *lb, MPI_Count *extent);
int MPI_Type_get_true_extent_x(MPI_Datatype datatype, MPI_Count *lb, MPI_Count *extent);
int MPI_Type_size_x(MPI_Datatype datatype, MPI_Count *size);

/* Noncollective communicator creation */
int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm);

/* MPI_Aint addressing arithmetic */
MPI_Aint MPI_Aint_add(MPI_Aint base, MPI_Aint disp);
MPI_Aint MPI_Aint_diff(MPI_Aint addr1, MPI_Aint addr2);

/* MPI_T interface */
/* The MPI_T routines are available only in C bindings - tell tools that they
   can skip these prototypes */
/* Begin Skip Prototypes */
int MPI_T_init_thread(int required, int *provided);
int MPI_T_finalize(void);
int MPI_T_enum_get_info(MPI_T_enum enumtype, int *num, char *name, int *name_len);
int MPI_T_enum_get_item(MPI_T_enum enumtype, int indx, int *value, char *name, int *name_len);
int MPI_T_cvar_get_num(int *num_cvar);
int MPI_T_cvar_get_info(int cvar_index, char *name, int *name_len, int *verbosity,
                        MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len,
                        int *binding, int *scope);
int MPI_T_cvar_handle_alloc(int cvar_index, void *obj_handle, MPI_T_cvar_handle *handle,
                            int *count);
int MPI_T_cvar_handle_free(MPI_T_cvar_handle *handle);
int MPI_T_cvar_read(MPI_T_cvar_handle handle, void *buf);
int MPI_T_cvar_write(MPI_T_cvar_handle handle, const void *buf);
int MPI_T_pvar_get_num(int *num_pvar);
int MPI_T_pvar_get_info(int pvar_index, char *name, int *name_len, int *verbosity, int *var_class,
                        MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len,
                        int *binding, int *readonly, int *continuous, int *atomic);
int MPI_T_pvar_session_create(MPI_T_pvar_session *session);
int MPI_T_pvar_session_free(MPI_T_pvar_session *session);
int MPI_T_pvar_handle_alloc(MPI_T_pvar_session session, int pvar_index, void *obj_handle,
                            MPI_T_pvar_handle *handle, int *count);
int MPI_T_pvar_handle_free(MPI_T_pvar_session session, MPI_T_pvar_handle *handle);
int MPI_T_pvar_start(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int MPI_T_pvar_stop(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int MPI_T_pvar_read(MPI_T_pvar_session session, MPI_T_pvar_handle handle, void *buf);
int MPI_T_pvar_write(MPI_T_pvar_session session, MPI_T_pvar_handle handle, const void *buf);
int MPI_T_pvar_reset(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int MPI_T_pvar_readreset(MPI_T_pvar_session session, MPI_T_pvar_handle handle, void *buf);
int MPI_T_category_get_num(int *num_cat);
int MPI_T_category_get_info(int cat_index, char *name, int *name_len, char *desc, int *desc_len,
                            int *num_cvars, int *num_pvars, int *num_categories);
int MPI_T_category_get_cvars(int cat_index, int len, int indices[]);
int MPI_T_category_get_pvars(int cat_index, int len, int indices[]);
int MPI_T_category_get_categories(int cat_index, int len, int indices[]);
int MPI_T_category_changed(int *stamp);
int MPI_T_cvar_get_index(const char *name, int *cvar_index);
int MPI_T_pvar_get_index(const char *name, int var_class, int *pvar_index);
int MPI_T_category_get_index(const char *name, int *cat_index);
/* End Skip Prototypes */


/* Non-standard but public extensions to MPI */
/* Fault Tolerance Extensions */
int MPIX_Comm_failure_ack(MPI_Comm comm);
int MPIX_Comm_failure_get_acked(MPI_Comm comm, MPI_Group *failedgrp);
int MPIX_Comm_revoke(MPI_Comm comm);
int MPIX_Comm_shrink(MPI_Comm comm, MPI_Comm *newcomm);
int MPIX_Comm_agree(MPI_Comm comm, int *flag);


/* End Prototypes */
#endif /* MPICH_SUPPRESS_PROTOTYPES */


/* Here are the bindings of the profiling routines */
#if !defined(MPI_BUILD_PROFILING)
int PMPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int PMPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Buffer_attach(void *buffer, int size);
int PMPI_Buffer_detach(void *buffer_addr, int *size);
int PMPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
               MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Wait(MPI_Request *request, MPI_Status *status);
int PMPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
int PMPI_Request_free(MPI_Request *request);
int PMPI_Waitany(int count, MPI_Request array_of_requests[], int *indx, MPI_Status *status);
int PMPI_Testany(int count, MPI_Request array_of_requests[], int *indx, int *flag,
                 MPI_Status *status);
int PMPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);
int PMPI_Testall(int count, MPI_Request array_of_requests[], int *flag,
                 MPI_Status array_of_statuses[]);
int PMPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount,
                  int array_of_indices[], MPI_Status array_of_statuses[]);
int PMPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount,
                  int array_of_indices[], MPI_Status array_of_statuses[]);
int PMPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
int PMPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int PMPI_Cancel(MPI_Request *request);
int PMPI_Test_cancelled(const MPI_Status *status, int *flag);
int PMPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                   MPI_Comm comm, MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Start(MPI_Request *request);
int PMPI_Startall(int count, MPI_Request array_of_requests[]);
int PMPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
                  int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int source, int recvtag, MPI_Comm comm, MPI_Status *status)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(6,8);
int PMPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest,
                          int sendtag, int source, int recvtag, MPI_Comm comm,
                          MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int PMPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype,
                     MPI_Datatype *newtype);
int PMPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype,
                      MPI_Datatype *newtype);
int PMPI_Type_indexed(int count, const int *array_of_blocklengths,
                      const int *array_of_displacements, MPI_Datatype oldtype,
                      MPI_Datatype *newtype);
int PMPI_Type_hindexed(int count, const int *array_of_blocklengths,
                       const MPI_Aint *array_of_displacements, MPI_Datatype oldtype,
                       MPI_Datatype *newtype);
int PMPI_Type_struct(int count, const int *array_of_blocklengths,
                     const MPI_Aint *array_of_displacements,
                     const MPI_Datatype *array_of_types, MPI_Datatype *newtype);
int PMPI_Address(const void *location, MPI_Aint *address);
int PMPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
int PMPI_Type_size(MPI_Datatype datatype, int *size);
int PMPI_Type_lb(MPI_Datatype datatype, MPI_Aint *displacement);
int PMPI_Type_ub(MPI_Datatype datatype, MPI_Aint *displacement);
int PMPI_Type_commit(MPI_Datatype *datatype);
int PMPI_Type_free(MPI_Datatype *datatype);
int PMPI_Get_elements(const MPI_Status *status, MPI_Datatype datatype, int *count);
int PMPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf,
              int outsize, int *position, MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Unpack(const void *inbuf, int insize, int *position, void *outbuf, int outcount,
                MPI_Datatype datatype, MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size);
int PMPI_Barrier(MPI_Comm comm);
int PMPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
               MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
                MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root,
                 MPI_Comm comm)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int PMPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,7);
int PMPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                    const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)
                    MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int PMPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls,
                   MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
                   const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,8);
int PMPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[],
                   const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                   const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm);
int PMPI_Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, MPI_Comm comm)
                MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, int root, MPI_Comm comm)
                MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op);
int PMPI_Op_free(MPI_Op *op);
int PMPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                   MPI_Op op, MPI_Comm comm)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[],
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
                        MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
              MPI_Comm comm)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Group_size(MPI_Group group, int *size);
int PMPI_Group_rank(MPI_Group group, int *rank);
int PMPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[], MPI_Group group2,
                               int ranks2[]);
int PMPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result);
int PMPI_Comm_group(MPI_Comm comm, MPI_Group *group);
int PMPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int PMPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int PMPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int PMPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);
int PMPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);
int PMPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
int PMPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
int PMPI_Group_free(MPI_Group *group);
int PMPI_Comm_size(MPI_Comm comm, int *size);
int PMPI_Comm_rank(MPI_Comm comm, int *rank);
int PMPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result);
int PMPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int PMPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm);
int PMPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
int PMPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
int PMPI_Comm_free(MPI_Comm *comm);
int PMPI_Comm_test_inter(MPI_Comm comm, int *flag);
int PMPI_Comm_remote_size(MPI_Comm comm, int *size);
int PMPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
int PMPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm,
                          int remote_leader, int tag, MPI_Comm *newintercomm);
int PMPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm);
int PMPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                       int *keyval, void *extra_state);
int PMPI_Keyval_free(int *keyval);
int PMPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val);
int PMPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
int PMPI_Attr_delete(MPI_Comm comm, int keyval);
int PMPI_Topo_test(MPI_Comm comm, int *status);
int PMPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[],
                     int reorder, MPI_Comm *comm_cart);
int PMPI_Dims_create(int nnodes, int ndims, int dims[]);
int PMPI_Graph_create(MPI_Comm comm_old, int nnodes, const int indx[], const int edges[],
                      int reorder, MPI_Comm *comm_graph);
int PMPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges);
int PMPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int indx[], int edges[]);
int PMPI_Cartdim_get(MPI_Comm comm, int *ndims);
int PMPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int PMPI_Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int PMPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int PMPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
int PMPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[]);
int PMPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest);
int PMPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm);
int PMPI_Cart_map(MPI_Comm comm, int ndims, const int dims[], const int periods[], int *newrank);
int PMPI_Graph_map(MPI_Comm comm, int nnodes, const int indx[], const int edges[], int *newrank);
int PMPI_Get_processor_name(char *name, int *resultlen);
int PMPI_Get_version(int *version, int *subversion);
int PMPI_Get_library_version(char *version, int *resultlen);
int PMPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler);
int PMPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int PMPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler);
int PMPI_Errhandler_free(MPI_Errhandler *errhandler);
int PMPI_Error_string(int errorcode, char *string, int *resultlen);
int PMPI_Error_class(int errorcode, int *errorclass);
double PMPI_Wtime(void);
double PMPI_Wtick(void);
int PMPI_Init(int *argc, char ***argv);
int PMPI_Finalize(void);
int PMPI_Initialized(int *flag);
int PMPI_Abort(MPI_Comm comm, int errorcode);

/* Note that we may need to define a @PCONTROL_LIST@ depending on whether
   stdargs are supported */
int PMPI_Pcontrol(const int level, ...);

/* Process Creation and Management */
int PMPI_Close_port(const char *port_name);
int PMPI_Comm_accept(const char *port_name, MPI_Info info, int root, MPI_Comm comm,
                     MPI_Comm *newcomm);
int PMPI_Comm_connect(const char *port_name, MPI_Info info, int root, MPI_Comm comm,
                      MPI_Comm *newcomm);
int PMPI_Comm_disconnect(MPI_Comm *comm);
int PMPI_Comm_get_parent(MPI_Comm *parent);
int PMPI_Comm_join(int fd, MPI_Comm *intercomm);
int PMPI_Comm_spawn(const char *command, char *argv[], int maxprocs, MPI_Info info, int root,
                    MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]);
int PMPI_Comm_spawn_multiple(int count, char *array_of_commands[], char **array_of_argv[],
                             const int array_of_maxprocs[], const MPI_Info array_of_info[],
                             int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]);
int PMPI_Lookup_name(const char *service_name, MPI_Info info, char *port_name);
int PMPI_Open_port(MPI_Info info, char *port_name);
int PMPI_Publish_name(const char *service_name, MPI_Info info, const char *port_name);
int PMPI_Unpublish_name(const char *service_name, MPI_Info info, const char *port_name);
int PMPI_Comm_set_info(MPI_Comm comm, MPI_Info info);
int PMPI_Comm_get_info(MPI_Comm comm, MPI_Info *info);

/* One-Sided Communications */
int PMPI_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                    int target_rank, MPI_Aint target_disp, int target_count,
                    MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
                    MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
             int target_rank, MPI_Aint target_disp, int target_count,
             MPI_Datatype target_datatype, MPI_Win win) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
             int target_rank, MPI_Aint target_disp, int target_count,
             MPI_Datatype target_datatype, MPI_Win win) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Win_complete(MPI_Win win);
int PMPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
                    MPI_Win *win);
int PMPI_Win_fence(int assert, MPI_Win win);
int PMPI_Win_free(MPI_Win *win);
int PMPI_Win_get_group(MPI_Win win, MPI_Group *group);
int PMPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win);
int PMPI_Win_post(MPI_Group group, int assert, MPI_Win win);
int PMPI_Win_start(MPI_Group group, int assert, MPI_Win win);
int PMPI_Win_test(MPI_Win win, int *flag);
int PMPI_Win_unlock(int rank, MPI_Win win);
int PMPI_Win_wait(MPI_Win win);

/* MPI-3 One-Sided Communication Routines */
int PMPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr,
                      MPI_Win *win);
int PMPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
                             void *baseptr, MPI_Win *win);
int PMPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr);
int PMPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win);
int PMPI_Win_attach(MPI_Win win, void *base, MPI_Aint size);
int PMPI_Win_detach(MPI_Win win, const void *base);
int PMPI_Win_get_info(MPI_Win win, MPI_Info *info_used);
int PMPI_Win_set_info(MPI_Win win, MPI_Info info);
int PMPI_Get_accumulate(const void *origin_addr, int origin_count,
                         MPI_Datatype origin_datatype, void *result_addr, int result_count,
                         MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                         int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
                         MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                         MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                       MPI_Datatype datatype, int target_rank, MPI_Aint target_disp,
                       MPI_Op op, MPI_Win win)
                       MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                           void *result_addr, MPI_Datatype datatype, int target_rank,
                           MPI_Aint target_disp, MPI_Win win)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,4);
int PMPI_Rput(const void *origin_addr, int origin_count,
               MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
               int target_count, MPI_Datatype target_datatype, MPI_Win win,
               MPI_Request *request)
               MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Rget(void *origin_addr, int origin_count,
               MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
               int target_count, MPI_Datatype target_datatype, MPI_Win win,
               MPI_Request *request)
               MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Raccumulate(const void *origin_addr, int origin_count,
                      MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
                      int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
                      MPI_Request *request)
                      MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Rget_accumulate(const void *origin_addr, int origin_count,
                          MPI_Datatype origin_datatype, void *result_addr, int result_count,
                          MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                          int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
                          MPI_Request *request)
                          MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                          MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Win_lock_all(int assert, MPI_Win win);
int PMPI_Win_unlock_all(MPI_Win win);
int PMPI_Win_flush(int rank, MPI_Win win);
int PMPI_Win_flush_all(MPI_Win win);
int PMPI_Win_flush_local(int rank, MPI_Win win);
int PMPI_Win_flush_local_all(MPI_Win win);
int PMPI_Win_sync(MPI_Win win);
 
/* External Interfaces */
int PMPI_Add_error_class(int *errorclass);
int PMPI_Add_error_code(int errorclass, int *errorcode);
int PMPI_Add_error_string(int errorcode, const char *string);
int PMPI_Comm_call_errhandler(MPI_Comm comm, int errorcode);
int PMPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn,
                            MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval,
                            void *extra_state);
int PMPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval);
int PMPI_Comm_free_keyval(int *comm_keyval);
int PMPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag);
int PMPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen);
int PMPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val);
int PMPI_Comm_set_name(MPI_Comm comm, const char *comm_name);
int PMPI_File_call_errhandler(MPI_File fh, int errorcode);
int PMPI_Grequest_complete(MPI_Request request);
int PMPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn,
                        MPI_Grequest_cancel_function *cancel_fn, void *extra_state,
                        MPI_Request *request);
int PMPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int PMPI_Is_thread_main(int *flag);
int PMPI_Query_thread(int *provided);
int PMPI_Status_set_cancelled(MPI_Status *status, int flag);
int PMPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count);
int PMPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn,
                            MPI_Type_delete_attr_function *type_delete_attr_fn,
                            int *type_keyval, void *extra_state);
int PMPI_Type_delete_attr(MPI_Datatype datatype, int type_keyval);
int PMPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype *newtype);
int PMPI_Type_free_keyval(int *type_keyval);
int PMPI_Type_get_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val, int *flag);
int PMPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses,
                           int max_datatypes, int array_of_integers[],
                           MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[]);
int PMPI_Type_get_envelope(MPI_Datatype datatype, int *num_integers, int *num_addresses,
                           int *num_datatypes, int *combiner);
int PMPI_Type_get_name(MPI_Datatype datatype, char *type_name, int *resultlen);
int PMPI_Type_set_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val);
int PMPI_Type_set_name(MPI_Datatype datatype, const char *type_name);
int PMPI_Type_match_size(int typeclass, int size, MPI_Datatype *datatype);
int PMPI_Win_call_errhandler(MPI_Win win, int errorcode);
int PMPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn,
                           MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval,
                           void *extra_state);
int PMPI_Win_delete_attr(MPI_Win win, int win_keyval);
int PMPI_Win_free_keyval(int *win_keyval);
int PMPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag);
int PMPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen);
int PMPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val);
int PMPI_Win_set_name(MPI_Win win, const char *win_name);

int PMPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr);
int PMPI_Comm_create_errhandler(MPI_Comm_errhandler_function *comm_errhandler_fn,
                                MPI_Errhandler *errhandler);
int PMPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler);
int PMPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler);
int PMPI_File_create_errhandler(MPI_File_errhandler_function *file_errhandler_fn,
                                MPI_Errhandler *errhandler);
int PMPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler);
int PMPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler);
int PMPI_Finalized(int *flag);
int PMPI_Free_mem(void *base);
int PMPI_Get_address(const void *location, MPI_Aint *address);
int PMPI_Info_create(MPI_Info *info);
int PMPI_Info_delete(MPI_Info info, const char *key);
int PMPI_Info_dup(MPI_Info info, MPI_Info *newinfo);
int PMPI_Info_free(MPI_Info *info);
int PMPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag);
int PMPI_Info_get_nkeys(MPI_Info info, int *nkeys);
int PMPI_Info_get_nthkey(MPI_Info info, int n, char *key);
int PMPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag);
int PMPI_Info_set(MPI_Info info, const char *key, const char *value);
int PMPI_Pack_external(const char datarep[], const void *inbuf, int incount,
                       MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position)
                       MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Pack_external_size(const char datarep[], int incount, MPI_Datatype datatype,
                            MPI_Aint *size);
int PMPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status);
int PMPI_Status_c2f(const MPI_Status *c_status, MPI_Fint *f_status);
int PMPI_Status_f2c(const MPI_Fint *f_status, MPI_Status *c_status);
int PMPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[],
                            const int array_of_distribs[], const int array_of_dargs[],
                            const int array_of_psizes[], int order, MPI_Datatype oldtype,
                            MPI_Datatype *newtype);
int PMPI_Type_create_hindexed(int count, const int array_of_blocklengths[],
                              const MPI_Aint array_of_displacements[], MPI_Datatype oldtype,
                              MPI_Datatype *newtype);
int PMPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype,
                             MPI_Datatype *newtype);
int PMPI_Type_create_indexed_block(int count, int blocklength, const int array_of_displacements[],
                                   MPI_Datatype oldtype, MPI_Datatype *newtype);
int PMPI_Type_create_hindexed_block(int count, int blocklength,
                                    const MPI_Aint array_of_displacements[],
                                    MPI_Datatype oldtype, MPI_Datatype *newtype);
int PMPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
                             MPI_Datatype *newtype);
int PMPI_Type_create_struct(int count, const int array_of_blocklengths[],
                            const MPI_Aint array_of_displacements[],
                            const MPI_Datatype array_of_types[], MPI_Datatype *newtype);
int PMPI_Type_create_subarray(int ndims, const int array_of_sizes[],
                              const int array_of_subsizes[], const int array_of_starts[],
                              int order, MPI_Datatype oldtype, MPI_Datatype *newtype);
int PMPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent);
int PMPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent);
int PMPI_Unpack_external(const char datarep[], const void *inbuf, MPI_Aint insize,
                         MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype)
                         MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,7);
int PMPI_Win_create_errhandler(MPI_Win_errhandler_function *win_errhandler_fn,
                               MPI_Errhandler *errhandler);
int PMPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler);
int PMPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler);

/* Fortran 90-related functions.  These routines are available only if
   Fortran 90 support is enabled 
*/
int PMPI_Type_create_f90_integer(int r, MPI_Datatype *newtype);
int PMPI_Type_create_f90_real(int p, int r, MPI_Datatype *newtype);
int PMPI_Type_create_f90_complex(int p, int r, MPI_Datatype *newtype);

int PMPI_Reduce_local(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype,
                      MPI_Op op)
                      MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Op_commutative(MPI_Op op, int *commute);
int PMPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
                              MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                              MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[],
                                    const int sourceweights[], int outdegree,
                                    const int destinations[], const int destweights[],
                                    MPI_Info info, int reorder, MPI_Comm *comm_dist_graph);
int PMPI_Dist_graph_create(MPI_Comm comm_old, int n, const int sources[], const int degrees[],
                           const int destinations[], const int weights[], MPI_Info info,
                           int reorder, MPI_Comm *comm_dist_graph);
int PMPI_Dist_graph_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree, int *weighted);
int PMPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[],
                              int maxoutdegree, int destinations[], int destweights[]);

/* Matched probe functionality */
int PMPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message,
                 MPI_Status *status);
int PMPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message,
                MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status);
int PMPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message,
               MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);

/* Nonblocking collectives */
int PMPI_Comm_idup(MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *request);
int PMPI_Ibarrier(MPI_Comm comm, MPI_Request *request);
int PMPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm,
                MPI_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);
int PMPI_Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
                 MPI_Request *request)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root,
                  MPI_Comm comm, MPI_Request *request)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int PMPI_Iscatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
                  MPI_Request *request)
                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Iscatterv(const void *sendbuf, const int sendcounts[], const int displs[],
                   MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   int root, MPI_Comm comm, MPI_Request *request)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,7);
int PMPI_Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                    int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                    MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                     const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                     MPI_Comm comm, MPI_Request *request)
                     MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int PMPI_Ialltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3) MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Ialltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                    MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                    const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                    MPI_Request *request)
                    MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,8);
int PMPI_Ialltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[],
                    const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                    const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm,
                    MPI_Request *request);
int PMPI_Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                 MPI_Op op, int root, MPI_Comm comm, MPI_Request *request)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Iallreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    MPI_Op op, MPI_Comm comm, MPI_Request *request)
                    MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Ireduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[],
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)
                         MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                         MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount,
                               MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                               MPI_Request *request)
                               MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                               MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Iscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
               MPI_Comm comm, MPI_Request *request)
               MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_Iexscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                 MPI_Op op, MPI_Comm comm, MPI_Request *request)
                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);

/* Neighborhood collectives */
int PMPI_Ineighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPI_Comm comm, MPI_Request *request)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Ineighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                              void *recvbuf, const int recvcounts[], const int displs[],
                              MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                              MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                              MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int PMPI_Ineighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm,
                            MPI_Request *request)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Ineighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                             MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                             const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                             MPI_Request *request)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,8);
int PMPI_Ineighbor_alltoallw(const void *sendbuf, const int sendcounts[],
                             const MPI_Aint sdispls[], const MPI_Datatype sendtypes[],
                             void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[],
                             const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request);
int PMPI_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Neighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, const int recvcounts[], const int displs[],
                             MPI_Datatype recvtype, MPI_Comm comm)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,7);
int PMPI_Neighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(4,6);
int PMPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                            MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,4)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(5,8);
int PMPI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[],
                            const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                            const MPI_Aint rdispls[], const MPI_Datatype recvtypes[],
                            MPI_Comm comm);

/* Shared memory */
int PMPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm);

/* Noncollective communicator creation */
int PMPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm);

/* MPI-3 "large count" routines */
int PMPI_Get_elements_x(const MPI_Status *status, MPI_Datatype datatype, MPI_Count *count);
int PMPI_Status_set_elements_x(MPI_Status *status, MPI_Datatype datatype, MPI_Count count);
int PMPI_Type_get_extent_x(MPI_Datatype datatype, MPI_Count *lb, MPI_Count *extent);
int PMPI_Type_get_true_extent_x(MPI_Datatype datatype, MPI_Count *lb, MPI_Count *extent);
int PMPI_Type_size_x(MPI_Datatype datatype, MPI_Count *size);

/* MPI_Aint addressing arithmetic */
MPI_Aint PMPI_Aint_add(MPI_Aint base, MPI_Aint disp);
MPI_Aint PMPI_Aint_diff(MPI_Aint addr1, MPI_Aint addr2);

/* MPI_T interface */
/* The MPI_T routines are available only in C bindings - tell tools that they
   can skip these prototypes */
/* Begin Skip Prototypes */
int PMPI_T_init_thread(int required, int *provided);
int PMPI_T_finalize(void);
int PMPI_T_enum_get_info(MPI_T_enum enumtype, int *num, char *name, int *name_len);
int PMPI_T_enum_get_item(MPI_T_enum enumtype, int indx, int *value, char *name, int *name_len);
int PMPI_T_cvar_get_num(int *num_cvar);
int PMPI_T_cvar_get_info(int cvar_index, char *name, int *name_len, int *verbosity,
                         MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len,
                         int *binding, int *scope);
int PMPI_T_cvar_handle_alloc(int cvar_index, void *obj_handle, MPI_T_cvar_handle *handle,
                             int *count);
int PMPI_T_cvar_handle_free(MPI_T_cvar_handle *handle);
int PMPI_T_cvar_read(MPI_T_cvar_handle handle, void *buf);
int PMPI_T_cvar_write(MPI_T_cvar_handle handle, const void *buf);
int PMPI_T_pvar_get_num(int *num_pvar);
int PMPI_T_pvar_get_info(int pvar_index, char *name, int *name_len, int *verbosity, int *var_class,
                         MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len,
                         int *binding, int *readonly, int *continuous, int *atomic);
int PMPI_T_pvar_session_create(MPI_T_pvar_session *session);
int PMPI_T_pvar_session_free(MPI_T_pvar_session *session);
int PMPI_T_pvar_handle_alloc(MPI_T_pvar_session session, int pvar_index, void *obj_handle,
                             MPI_T_pvar_handle *handle, int *count);
int PMPI_T_pvar_handle_free(MPI_T_pvar_session session, MPI_T_pvar_handle *handle);
int PMPI_T_pvar_start(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int PMPI_T_pvar_stop(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int PMPI_T_pvar_read(MPI_T_pvar_session session, MPI_T_pvar_handle handle, void *buf);
int PMPI_T_pvar_write(MPI_T_pvar_session session, MPI_T_pvar_handle handle, const void *buf);
int PMPI_T_pvar_reset(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int PMPI_T_pvar_readreset(MPI_T_pvar_session session, MPI_T_pvar_handle handle, void *buf);
int PMPI_T_category_get_num(int *num_cat);
int PMPI_T_category_get_info(int cat_index, char *name, int *name_len, char *desc, int *desc_len,
                             int *num_cvars, int *num_pvars, int *num_categories);
int PMPI_T_category_get_cvars(int cat_index, int len, int indices[]);
int PMPI_T_category_get_pvars(int cat_index, int len, int indices[]);
int PMPI_T_category_get_categories(int cat_index, int len, int indices[]);
int PMPI_T_category_changed(int *stamp);
int PMPI_T_cvar_get_index(const char *name, int *cvar_index);
int PMPI_T_pvar_get_index(const char *name, int var_class, int *pvar_index);
int PMPI_T_category_get_index(const char *name, int *cat_index);
/* End Skip Prototypes */


/* Non-standard but public extensions to MPI */
/* Fault Tolerance Extensions */
int PMPIX_Comm_failure_ack(MPI_Comm comm);
int PMPIX_Comm_failure_get_acked(MPI_Comm comm, MPI_Group *failedgrp);
int PMPIX_Comm_revoke(MPI_Comm comm);
int PMPIX_Comm_shrink(MPI_Comm comm, MPI_Comm *newcomm);
int PMPIX_Comm_agree(MPI_Comm comm, int *flag);

#endif  /* MPI_BUILD_PROFILING */
/* End of MPI bindings */

/* feature advertisement */
#define MPIIMPL_ADVERTISES_FEATURES 1
#define MPIIMPL_HAVE_MPI_INFO 1                                                 
#define MPIIMPL_HAVE_MPI_COMBINER_DARRAY 1                                      
#define MPIIMPL_HAVE_MPI_TYPE_CREATE_DARRAY 1
#define MPIIMPL_HAVE_MPI_COMBINER_SUBARRAY 1                                    
#define MPIIMPL_HAVE_MPI_TYPE_CREATE_DARRAY 1
#define MPIIMPL_HAVE_MPI_COMBINER_DUP 1                                         
#define MPIIMPL_HAVE_MPI_GREQUEST 1      
#define MPIIMPL_HAVE_STATUS_SET_BYTES 1
#define MPIIMPL_HAVE_STATUS_SET_INFO 1

#include "mpio.h"

#if defined(__cplusplus)
}
/* Add the C++ bindings */
/* 
   If MPICH_SKIP_MPICXX is defined, the mpicxx.h file will *not* be included.
   This is necessary, for example, when building the C++ interfaces.  It
   can also be used when you want to use a C++ compiler to compile C code,
   and do not want to load the C++ bindings.  These definitions can
   be made by the C++ compilation script
 */
#if !defined(MPICH_SKIP_MPICXX)
/* mpicxx.h contains the MPI C++ binding.  In the mpi.h.in file, this 
   include is in an autoconf variable in case the compiler is a C++ 
   compiler but MPI was built without the C++ bindings */
#include "mpicxx.h"
#endif 
#endif


/* Generalized requests extensions */
typedef int MPIX_Grequest_class;
int MPIX_Grequest_class_create(MPI_Grequest_query_function *query_fn,
                               MPI_Grequest_free_function *free_fn,
                               MPI_Grequest_cancel_function *cancel_fn,
                               MPIX_Grequest_poll_function *poll_fn,
                               MPIX_Grequest_wait_function *wait_fn,
                               MPIX_Grequest_class *greq_class);
int MPIX_Grequest_class_allocate(MPIX_Grequest_class greq_class, void *extra_state,
                                 MPI_Request *request);
int MPIX_Grequest_start(MPI_Grequest_query_function *query_fn,
                        MPI_Grequest_free_function *free_fn,
                        MPI_Grequest_cancel_function *cancel_fn,
                        MPIX_Grequest_poll_function *poll_fn,
                        MPIX_Grequest_wait_function *wait_fn, void *extra_state,
                        MPI_Request *request);

/* RMA Mutexes Extensions */
struct mpixi_mutex_s;
typedef struct mpixi_mutex_s * MPIX_Mutex;
int MPIX_Mutex_create(int count, MPI_Comm comm, MPIX_Mutex *hdl);
int MPIX_Mutex_free(MPIX_Mutex *hdl);
int MPIX_Mutex_lock(MPIX_Mutex hdl, int mutex, int proc);
int MPIX_Mutex_unlock(MPIX_Mutex hdl, int mutex, int proc);


#if !defined(MPI_BUILD_PROFILING)
/* Generalized requests extensions */
int PMPIX_Grequest_class_create(MPI_Grequest_query_function *query_fn,
                                MPI_Grequest_free_function *free_fn,
                                MPI_Grequest_cancel_function *cancel_fn,
                                MPIX_Grequest_poll_function *poll_fn,
                                MPIX_Grequest_wait_function *wait_fn,
                                MPIX_Grequest_class *greq_class);
int PMPIX_Grequest_class_allocate(MPIX_Grequest_class greq_class, void *extra_state,
                                  MPI_Request *request);
int PMPIX_Grequest_start(MPI_Grequest_query_function *query_fn,
                         MPI_Grequest_free_function *free_fn,
                         MPI_Grequest_cancel_function *cancel_fn,
                         MPIX_Grequest_poll_function *poll_fn,
                         MPIX_Grequest_wait_function *wait_fn, void *extra_state,
                         MPI_Request *request);

/* RMA Mutexes Extensions */
int PMPIX_Mutex_create(int count, MPI_Comm comm, MPIX_Mutex *hdl);
int PMPIX_Mutex_free(MPIX_Mutex *hdl);
int PMPIX_Mutex_lock(MPIX_Mutex hdl, int mutex, int proc);
int PMPIX_Mutex_unlock(MPIX_Mutex hdl, int mutex, int proc);
#endif /* MPI_BUILD_PROFILING */

#endif
