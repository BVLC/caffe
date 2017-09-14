/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* 
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

/* user include file for MPI-IO programs */

#ifndef MPIO_INCLUDE
#define MPIO_INCLUDE

#include "mpi.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define ROMIO_VERSION 126 /* version 1.2.6 */

/* define MPI-IO datatypes and constants */

#ifndef MPI_FILE_DEFINED
typedef struct ADIOI_FileD *MPI_File;
#endif

#define HAVE_MPI_GREQUEST 1
#ifndef HAVE_MPI_GREQUEST
typedef struct ADIOI_RequestD *MPIO_Request;  
#else
#define MPIO_Request MPI_Request
#define MPIO_USES_MPI_REQUEST
/* Also rename the MPIO routines to get the MPI versions */
#define MPIO_Wait MPI_Wait
#define MPIO_Test MPI_Test
#define PMPIO_Wait PMPI_Wait
#define PMPIO_Test PMPI_Test
#endif
#define MPIO_REQUEST_DEFINED

#ifndef HAVE_MPI_OFFSET
typedef long long MPI_Offset;
/* If we needed to define MPI_Offset, then we also need to make
   this definition. */
#ifndef HAVE_MPI_DATAREP_FUNCTIONS
#define HAVE_MPI_DATAREP_FUNCTIONS
typedef int (MPI_Datarep_conversion_function)(void *, MPI_Datatype, int, 
             void *, MPI_Offset, void *);
typedef int (MPI_Datarep_extent_function)(MPI_Datatype datatype, MPI_Aint *,
					  void *);
#endif
#endif

#ifndef NEEDS_MPI_FINT

#endif
#ifdef NEEDS_MPI_FINT
typedef int MPI_Fint; 
#endif

#ifndef HAVE_MPI_INFO
#define HAVE_MPI_INFO
#endif
#ifndef HAVE_MPI_INFO
  typedef struct MPIR_Info *MPI_Info;
# define MPI_INFO_NULL         ((MPI_Info) 0)
# define MPI_MAX_INFO_KEY       255
# define MPI_MAX_INFO_VAL      1024
#endif

#define MPI_MODE_RDONLY              2  /* ADIO_RDONLY */
#define MPI_MODE_RDWR                8  /* ADIO_RDWR  */
#define MPI_MODE_WRONLY              4  /* ADIO_WRONLY  */
#define MPI_MODE_CREATE              1  /* ADIO_CREATE */ 
#define MPI_MODE_EXCL               64  /* ADIO_EXCL */
#define MPI_MODE_DELETE_ON_CLOSE    16  /* ADIO_DELETE_ON_CLOSE */
#define MPI_MODE_UNIQUE_OPEN        32  /* ADIO_UNIQUE_OPEN */
#define MPI_MODE_APPEND            128  /* ADIO_APPEND */
#define MPI_MODE_SEQUENTIAL        256  /* ADIO_SEQUENTIAL */

#define MPI_DISPLACEMENT_CURRENT   -54278278

#ifndef MPICH
/* FIXME: Make sure that we get a consistent definition of MPI_FILE_NULL
	in MPICH */
/* MPICH defines null object handles differently */
#define MPI_FILE_NULL           ((MPI_File) 0)
#endif
#define MPIO_REQUEST_NULL       ((MPIO_Request) 0)

#define MPI_SEEK_SET            600
#define MPI_SEEK_CUR            602
#define MPI_SEEK_END            604

/* Open MPI: don't define MPI_MAX_DATAREP_STRING here; it's defined in
   OMPI's mpi.h. */
#ifndef OPEN_MPI
#define MPI_MAX_DATAREP_STRING  128
#endif

#ifndef HAVE_MPI_DARRAY_SUBARRAY
#define HAVE_MPI_DARRAY_SUBARRAY
#endif
#ifndef HAVE_MPI_DARRAY_SUBARRAY
#  define MPI_ORDER_C             56
#  define MPI_ORDER_FORTRAN       57
#  define MPI_DISTRIBUTE_BLOCK    121
#  define MPI_DISTRIBUTE_CYCLIC   122
#  define MPI_DISTRIBUTE_NONE     123
#  define MPI_DISTRIBUTE_DFLT_DARG -49767
#endif


/* MPI-IO function prototypes */

/* The compiler must support ANSI C style prototypes, otherwise 
   long long constants (e.g. 0) may get passed as ints. */

#ifndef HAVE_PRAGMA_HP_SEC_DEF

/* Section 9.2 */
/* Begin Prototypes */
int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);
int MPI_File_close(MPI_File *fh);
int MPI_File_delete(const char *filename, MPI_Info info);
int MPI_File_set_size(MPI_File fh, MPI_Offset size);
int MPI_File_preallocate(MPI_File fh, MPI_Offset size);
int MPI_File_get_size(MPI_File fh, MPI_Offset *size);
int MPI_File_get_group(MPI_File fh, MPI_Group *group);
int MPI_File_get_amode(MPI_File fh, int *amode);
int MPI_File_set_info(MPI_File fh, MPI_Info info);
int MPI_File_get_info(MPI_File fh, MPI_Info *info_used);

/* Section 9.3 */
int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype,
                      const char *datarep, MPI_Info info);
int MPI_File_get_view(MPI_File fh, MPI_Offset *disp, MPI_Datatype *etype, MPI_Datatype *filetype,
                      char *datarep);

/* Section 9.4.2 */
int MPI_File_read_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype,
                     MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int MPI_File_read_at_all(MPI_File fh, MPI_Offset offset, void * buf, int count,
                         MPI_Datatype datatype, MPI_Status *status)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int MPI_File_write_at(MPI_File fh, MPI_Offset offset, const void * buf, int count,
                      MPI_Datatype datatype, MPI_Status *status)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int MPI_File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count,
                          MPI_Datatype datatype, MPI_Status *status)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);

/* nonblocking calls currently use MPIO_Request, because generalized
   requests not yet implemented. For the same reason, MPIO_Test and 
   MPIO_Wait are used to test and wait on nonblocking I/O requests */ 
int MPI_File_iread_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype,
                      MPIO_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int MPI_File_iwrite_at(MPI_File fh, MPI_Offset offset, const void *buf, int count,
                       MPI_Datatype datatype, MPIO_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);

/* Section 9.4.3 */
int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_read_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                   MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_write_all(MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                       MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);

/* nonblocking calls currently use MPIO_Request, because generalized
   requests not yet implemented. For the same reason, MPIO_Test and 
   MPIO_Wait are used to test and wait on nonblocking I/O requests */ 

int MPI_File_iread(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPIO_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_iwrite(MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                    MPIO_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);

int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence);
int MPI_File_get_position(MPI_File fh, MPI_Offset *offset);
int MPI_File_get_byte_offset(MPI_File fh, MPI_Offset offset, MPI_Offset *disp);

/* Section 9.4.4 */
int MPI_File_read_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                         MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_write_shared(MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                          MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_iread_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                          MPIO_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_iwrite_shared(MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                           MPIO_Request *request) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_read_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                          MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_write_ordered(MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                           MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_seek_shared(MPI_File fh, MPI_Offset offset, int whence);
int MPI_File_get_position_shared(MPI_File fh, MPI_Offset *offset);

/* Section 9.4.5 */
int MPI_File_read_at_all_begin(MPI_File fh, MPI_Offset offset, void *buf, int count,
                               MPI_Datatype datatype) MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int MPI_File_read_at_all_end(MPI_File fh, void *buf, MPI_Status *status);
int MPI_File_write_at_all_begin(MPI_File fh, MPI_Offset offset, const void *buf, int count,
                                MPI_Datatype datatype) MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int MPI_File_write_at_all_end(MPI_File fh, const void *buf, MPI_Status *status);
int MPI_File_read_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_read_all_end(MPI_File fh, void *buf, MPI_Status *status);
int MPI_File_write_all_begin(MPI_File fh, const void *buf, int count, MPI_Datatype datatype)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_write_all_end(MPI_File fh, const void *buf, MPI_Status *status);
int MPI_File_read_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_read_ordered_end(MPI_File fh, void *buf, MPI_Status *status);
int MPI_File_write_ordered_begin(MPI_File fh, const void *buf, int count, MPI_Datatype datatype)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_write_ordered_end(MPI_File fh, const void *buf, MPI_Status *status);

/* Section 9.5.1 */
int MPI_File_get_type_extent(MPI_File fh, MPI_Datatype datatype, MPI_Aint *extent);

/* Section 9.5.3 */
int MPI_Register_datarep(const char *datarep, MPI_Datarep_conversion_function *read_conversion_fn,
			 MPI_Datarep_conversion_function *write_conversion_fn,
			 MPI_Datarep_extent_function *dtype_file_extent_fn, void *extra_state);

/* Section 9.6.1 */
int MPI_File_set_atomicity(MPI_File fh, int flag);
int MPI_File_get_atomicity(MPI_File fh, int *flag);
int MPI_File_sync(MPI_File fh);

/* Section 4.13.3 */
#ifndef MPICH
/* MPICH provides these definitions */
int MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler);
int MPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler);
#endif

/* For MPI 3.1 */
int MPI_File_iread_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count,
                           MPI_Datatype datatype, MPI_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int MPI_File_iwrite_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count,
                            MPI_Datatype datatype, MPI_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int MPI_File_iread_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                        MPI_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int MPI_File_iwrite_all(MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                         MPI_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
/* End Prototypes */

#ifndef HAVE_MPI_DARRAY_SUBARRAY
/* Section 4.14.4 */
int MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[],
                             const int array_of_starts[], int order, MPI_Datatype oldtype,
                             MPI_Datatype *newtype);

/* Section 4.14.5 */
int MPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[],
                           const int array_of_distribs[], const int array_of_dargs[],
                           const int array_of_psizes, int order, MPI_Datatype oldtype,
                           MPI_Datatype *newtype);
#endif

/* The globus2 device has to rename MPI_ symbols in order to use the vendor
   MPI as one of its transport mechanisms.  Therefore, the following undefines
   should only happen if MPICH_RENAMING_MPI_FUNCS is not defined. */
/* Section 4.12.4 */
#if !defined(MPICH_RENAMING_MPI_FUNCS)
#ifdef MPI_File_f2c
#undef MPI_File_f2c
#endif
#ifdef MPI_File_c2f
#undef MPI_File_c2f
#endif
#endif
/* above needed for some versions of mpi.h in MPICH!! */
MPI_File MPI_File_f2c(MPI_Fint file);
MPI_Fint MPI_File_c2f(MPI_File file);


#ifndef HAVE_MPI_GREQUEST
/* The following functions are required if generalized requests are not
   available, because in that case, an MPIO_Request object
   is currently used for nonblocking I/O. */
int MPIO_Test(MPIO_Request *request, int *flag, MPI_Status *status);
int MPIO_Wait(MPIO_Request *request, MPI_Status *status);
int MPIO_Testall(int count, MPIO_Request array_of_requests[], int *flag,
                 MPI_Status array_of_statuses[]);
int MPIO_Waitall(int count, MPIO_Request array_of_requests[], MPI_Status array_of_statuses[]);
int MPIO_Testany(int count, MPIO_Request array_of_requests[], int *indx, int *flag,
                 MPI_Status *status);
int MPIO_Waitany(int count, MPIO_Request array_of_requests[], int *indx, MPI_Status *status);
int MPIO_Waitsome(int incount, MPIO_Request array_of_requests[], int *outcount,
                  int array_of_indices[], MPI_Status array_of_statuses[]);
int MPIO_Testsome(int incount, MPIO_Request array_of_requests[], int *outcount,
                  int array_of_indices[], MPI_Status array_of_statuses[]);

MPI_Fint MPIO_Request_c2f(MPIO_Request request);
MPIO_Request MPIO_Request_f2c(MPI_Fint request);
#endif /* HAVE_MPI_GREQUEST */

/* info functions if not defined in the MPI implementation */
#ifndef HAVE_MPI_INFO

int MPI_Info_create(MPI_Info *info);
int MPI_Info_set(MPI_Info info, const char *key, const char *value);
int MPI_Info_delete(MPI_Info info, const char *key);
int MPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag);
int MPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag);
int MPI_Info_get_nkeys(MPI_Info info, int *nkeys);
int MPI_Info_get_nthkey(MPI_Info info, int n, char *key);
int MPI_Info_dup(MPI_Info info, MPI_Info *newinfo);
int MPI_Info_free(MPI_Info *info);

/* The globus2 device has to rename MPI_ symbols in order to use the vendor
   MPI as one of its transport mechanisms.  Therefore, the following undefines
   should only happen if MPICH_RENAMING_MPI_FUNCS is not defined. */
#if !defined(MPICH_RENAMING_MPI_FUNCS)
#ifdef MPI_Info_f2c
#undef MPI_Info_f2c
#endif
#ifdef MPI_Info_c2f
#undef MPI_Info_c2f
#endif
#endif
/* above needed for some versions of mpi.h in MPICH!! */
MPI_Fint MPI_Info_c2f(MPI_Info info);
MPI_Info MPI_Info_f2c(MPI_Fint info);
#endif

#endif   /* HAVE_PRAGMA_HP_SEC_DEF */


/**************** BINDINGS FOR THE PROFILING INTERFACE ***************/


/* Section 9.2 */
int PMPI_File_open(MPI_Comm, const char *, int, MPI_Info, MPI_File *);
int PMPI_File_close(MPI_File *);
int PMPI_File_delete(const char *, MPI_Info);
int PMPI_File_set_size(MPI_File, MPI_Offset);
int PMPI_File_preallocate(MPI_File, MPI_Offset);
int PMPI_File_get_size(MPI_File, MPI_Offset *);
int PMPI_File_get_group(MPI_File, MPI_Group *);
int PMPI_File_get_amode(MPI_File, int *);
int PMPI_File_set_info(MPI_File, MPI_Info);
int PMPI_File_get_info(MPI_File, MPI_Info *);

/* Section 9.3 */
int PMPI_File_set_view(MPI_File, MPI_Offset, 
    MPI_Datatype, MPI_Datatype, const char *, MPI_Info);
int PMPI_File_get_view(MPI_File, MPI_Offset *, 
      MPI_Datatype *, MPI_Datatype *, char *);

/* Section 9.4.2 */
int PMPI_File_read_at(MPI_File, MPI_Offset, void *,
	      int, MPI_Datatype, MPI_Status *)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int PMPI_File_read_at_all(MPI_File, MPI_Offset, void *,
	      int, MPI_Datatype, MPI_Status *)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int PMPI_File_write_at(MPI_File, MPI_Offset, const void *,
	      int, MPI_Datatype, MPI_Status *)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int PMPI_File_write_at_all(MPI_File, MPI_Offset, const void *,
	      int, MPI_Datatype, MPI_Status *)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);

/* nonblocking calls currently use MPIO_Request, because generalized
   requests not yet implemented. For the same reason, MPIO_Test and 
   MPIO_Wait are used to test and wait on nonblocking I/O requests */ 

int PMPI_File_iread_at(MPI_File, MPI_Offset, void *,
	      int, MPI_Datatype, MPIO_Request *)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int PMPI_File_iwrite_at(MPI_File, MPI_Offset, const void *,
	      int, MPI_Datatype, MPIO_Request *)
              MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);

/* Section 9.4.3 */
int PMPI_File_read(MPI_File, void *, int, MPI_Datatype, MPI_Status *)
                   MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_read_all(MPI_File, void *, int, MPI_Datatype, MPI_Status *)
                       MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_write(MPI_File, const void *, int, MPI_Datatype, MPI_Status *)
                    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_write_all(MPI_File, const void *, int, MPI_Datatype, MPI_Status *)
                        MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);

/* nonblocking calls currently use MPIO_Request, because generalized
   requests not yet implemented. For the same reason, MPIO_Test and 
   MPIO_Wait are used to test and wait on nonblocking I/O requests */ 

int PMPI_File_iread(MPI_File, void *, int, MPI_Datatype, MPIO_Request *)
                    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_iwrite(MPI_File, const void *, int, MPI_Datatype, MPIO_Request *)
                     MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);

int PMPI_File_seek(MPI_File, MPI_Offset, int);
int PMPI_File_get_position(MPI_File, MPI_Offset *);
int PMPI_File_get_byte_offset(MPI_File, MPI_Offset, MPI_Offset *);

/* Section 9.4.4 */
int PMPI_File_read_shared(MPI_File, void *, int, MPI_Datatype, MPI_Status *)
                          MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_write_shared(MPI_File, const void *, int, MPI_Datatype, MPI_Status *)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_iread_shared(MPI_File, void *, int, 
			   MPI_Datatype, MPIO_Request *)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_iwrite_shared(MPI_File, const void *, int,
			    MPI_Datatype, MPIO_Request *)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_read_ordered(MPI_File, void *, int, MPI_Datatype, MPI_Status *)
                           MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_write_ordered(MPI_File, const void *, int, MPI_Datatype, MPI_Status *)
                            MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_seek_shared(MPI_File, MPI_Offset, int);
int PMPI_File_get_position_shared(MPI_File, MPI_Offset *);

/* Section 9.4.5 */
int PMPI_File_read_at_all_begin(MPI_File, MPI_Offset, void *,
                               int, MPI_Datatype)
                               MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int PMPI_File_read_at_all_end(MPI_File, void *, MPI_Status *);
int PMPI_File_write_at_all_begin(MPI_File, MPI_Offset, const void *,
                                 int, MPI_Datatype)
                                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int PMPI_File_write_at_all_end(MPI_File, const void *, MPI_Status *);
int PMPI_File_read_all_begin(MPI_File, void *, int, MPI_Datatype)
                             MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_read_all_end(MPI_File, void *, MPI_Status *);
int PMPI_File_write_all_begin(MPI_File, const void *, int, MPI_Datatype)
                              MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_write_all_end(MPI_File, const void *, MPI_Status *);
int PMPI_File_read_ordered_begin(MPI_File, void *, int, MPI_Datatype)
                                 MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_read_ordered_end(MPI_File, void *, MPI_Status *);
int PMPI_File_write_ordered_begin(MPI_File, const void *, int, MPI_Datatype)
                                  MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_write_ordered_end(MPI_File, const void *, MPI_Status *);

/* Section 9.5.1 */
int PMPI_File_get_type_extent(MPI_File, MPI_Datatype, MPI_Aint *);

/* Section 9.5.3 */
int PMPI_Register_datarep(const char *,
			 MPI_Datarep_conversion_function *,
			 MPI_Datarep_conversion_function *,
			 MPI_Datarep_extent_function *,
			 void *);

/* Section 9.6.1 */
int PMPI_File_set_atomicity(MPI_File, int);
int PMPI_File_get_atomicity(MPI_File, int *);
int PMPI_File_sync(MPI_File);

/* Section 4.13.3 */
#ifndef MPICH
/* MPICH provides these definitions */
int PMPI_File_set_errhandler( MPI_File, MPI_Errhandler );
int PMPI_File_get_errhandler( MPI_File, MPI_Errhandler * );
#endif

/* For MPI 3.1 */
int PMPI_File_iread_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count,
                            MPI_Datatype datatype, MPI_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int PMPI_File_iwrite_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count,
                             MPI_Datatype datatype, MPI_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5);
int PMPI_File_iread_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                         MPI_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);
int PMPI_File_iwrite_all(MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                          MPI_Request *request)
    MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4);

#ifndef HAVE_MPI_DARRAY_SUBARRAY
/* Section 4.14.4 */
int PMPI_Type_create_subarray(int, int *, int *, int *, int, 
                      MPI_Datatype, MPI_Datatype *);

/* Section 4.14.5 */
int PMPI_Type_create_darray(int, int, int, int *, int *, 
                    int *, int *, int, MPI_Datatype, MPI_Datatype *);
#endif

/* Section 4.12.4 */
MPI_File PMPI_File_f2c(MPI_Fint);
MPI_Fint PMPI_File_c2f(MPI_File);

#ifndef HAVE_MPI_GREQUEST
/* The following functions are required if generalized requests are not
   available, because in that case, an MPIO_Request object
   is currently used for nonblocking I/O. */
int PMPIO_Test(MPIO_Request *, int *, MPI_Status *);
int PMPIO_Wait(MPIO_Request *, MPI_Status *);
int PMPIO_Testall(int, MPIO_Request *, int *, MPI_Status *);
int PMPIO_Waitall(int, MPIO_Request *, MPI_Status *);
int PMPIO_Testany(int, MPIO_Request *, int *, int *, MPI_Status *);
int PMPIO_Waitany(int, MPIO_Request *, int *, MPI_Status *);
int PMPIO_Waitsome(int, MPIO_Request *, int *, int *, MPI_Status *);
int PMPIO_Testsome(int, MPIO_Request *, int *, int *, MPI_Status *);
MPI_Fint PMPIO_Request_c2f(MPIO_Request);
MPIO_Request PMPIO_Request_f2c(MPI_Fint);
#endif /* HAVE_MPI_GREQUEST */

/* info functions if not defined in the MPI implementation */
#ifndef HAVE_MPI_INFO

int PMPI_Info_create(MPI_Info *);
int PMPI_Info_set(MPI_Info, char *, char *);
int PMPI_Info_delete(MPI_Info, char *);
int PMPI_Info_get(MPI_Info, char *, int, char *, int *);
int PMPI_Info_get_valuelen(MPI_Info, char *, int *, int *);
int PMPI_Info_get_nkeys(MPI_Info, int *);
int PMPI_Info_get_nthkey(MPI_Info, int, char *);
int PMPI_Info_dup(MPI_Info, MPI_Info *);
int PMPI_Info_free(MPI_Info *);

MPI_Fint PMPI_Info_c2f(MPI_Info);
MPI_Info PMPI_Info_f2c(MPI_Fint);
#endif

#if defined(__cplusplus)
}
#endif

#endif
