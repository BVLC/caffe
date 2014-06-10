#!/usr/bin/env python

import sys, re, os.path
from string import Template

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

class_ignore_list = (
)

const_ignore_list = (
)

const_private_list = (
)

# { Module : { public : [[name, val],...], private : [[]...] } }
missing_consts = \
{
    'caffe' :
    {
        'private' :
        (
        ), # private
        'public' :
        (
        ) #public
    }， # caffe
}


# c_type    : { java/jni correspondence }
type_dict = {
# "simple"  : { j_type : "?", jn_type : "?", jni_type : "?", suffix : "?" },
    ""        : { "j_type" : "", "jn_type" : "long", "jni_type" : "jlong" }, # c-tor ret_type
    "void"    : { "j_type" : "void", "jn_type" : "void", "jni_type" : "void" },
    "env"     : { "j_type" : "", "jn_type" : "", "jni_type" : "JNIEnv*"},
    "cls"     : { "j_type" : "", "jn_type" : "", "jni_type" : "jclass"},
    "bool"    : { "j_type" : "boolean", "jn_type" : "boolean", "jni_type" : "jboolean", "suffix" : "Z" },
    "int"     : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "long"    : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "float"   : { "j_type" : "float", "jn_type" : "float", "jni_type" : "jfloat", "suffix" : "F" },
    "double"  : { "j_type" : "double", "jn_type" : "double", "jni_type" : "jdouble", "suffix" : "D" },
    "size_t"  : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "__int64" : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "int64"   : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "double[]": { "j_type" : "double[]", "jn_type" : "double[]", "jni_type" : "jdoubleArray", "suffix" : "_3D" },

# "complex" : { j_type : "?", jn_args : (("", ""),), jn_name : "", jni_var : "", jni_name : "", "suffix" : "?" },
}

# { class : { func : {j_code, jn_code, cpp_code} } }
ManualFuncs = {
}

# { class : { func : { arg_name : {"ctype" : ctype, "attrib" : [attrib]} } } }
func_arg_fix = {
    '' : {
    }, # '', i.e. no class
} # func_arg_fix


def getLibVersion(version_hpp_path):
    version_file = open(version_hpp_path, "rt").read()
    major = re.search("^W*#\W*define\W+CV_VERSION_MAJOR\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    minor = re.search("^W*#\W*define\W+CV_VERSION_MINOR\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    revision = re.search("^W*#\W*define\W+CV_VERSION_REVISION\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    status = re.search("^W*#\W*define\W+CV_VERSION_STATUS\W+\"(.*?)\"\W*$", version_file, re.MULTILINE).group(1)
    return (major, minor, revision, status)

class ConstInfo(object):
    def __init__(self, cname, name, val, addedManually=False):
        self.cname = cname
        self.name = re.sub(r"^Cv", "", name)
        self.value = val
        self.addedManually = addedManually


class ClassPropInfo(object):
    def __init__(self, decl): # [f_ctype, f_name, '', '/RW']
        self.ctype = decl[0]
        self.name = decl[1]
        self.rw = "/RW" in decl[3]

class ClassInfo(object):
    def __init__(self, decl): # [ 'class/struct cname', ': base', [modlist] ]
        name = decl[0]
        name = name[name.find(" ")+1:].strip()
        self.cname = self.name = self.jname = re.sub(r"^cv\.", "", name)
        self.cname = self.cname.replace(".", "::")
        self.methods = {}
        self.methods_suffixes = {}
        self.consts = [] # using a list to save the occurence order
        self.private_consts = []
        self.imports = set()
        self.props= []
        self.jname = self.name
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
        self.base = ''
        if decl[1]:
            #self.base = re.sub(r"\b"+self.jname+r"\b", "", decl[1].replace(":", "")).strip()
            self.base = re.sub(r"^.*:", "", decl[1].split(",")[0]).strip().replace(self.jname, "")

class ArgInfo(object):
    def __init__(self, arg_tuple): # [ ctype, name, def val, [mod], argno ]
        self.pointer = False
        ctype = arg_tuple[0]
        if ctype.endswith("*"):
            ctype = ctype[:-1]
            self.pointer = True
        if ctype == 'vector_Point2d':
            ctype = 'vector_Point2f'
        elif ctype == 'vector_Point3d':
            ctype = 'vector_Point3f'
        self.ctype = ctype
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.out = ""
        if "/O" in arg_tuple[3]:
            self.out = "O"
        if "/IO" in arg_tuple[3]:
            self.out = "IO"


class FuncInfo(object):
    def __init__(self, decl): # [ funcname, return_ctype, [modifiers], [args] ]
        name = re.sub(r"^cv\.", "", decl[0])
        self.cname = name.replace(".", "::")
        classname = ""
        dpos = name.rfind(".")
        if dpos >= 0:
            classname = name[:dpos]
            name = name[dpos+1:]
        self.classname = classname
        self.jname = self.name = name
        if "[" in name:
            self.jname = "getelem"
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
        self.static = ["","static"][ "/S" in decl[2] ]
        self.ctype = re.sub(r"^CvTermCriteria", "TermCriteria", decl[1] or "")
        self.args = []
        func_fix_map = func_arg_fix.get(classname, {}).get(self.jname, {})
        for a in decl[3]:
            arg = a[:]
            arg_fix_map = func_fix_map.get(arg[1], {})
            arg[0] = arg_fix_map.get('ctype',  arg[0]) #fixing arg type
            arg[3] = arg_fix_map.get('attrib', arg[3]) #fixing arg attrib
            ai = ArgInfo(arg)
            self.args.append(ai)



class FuncFamilyInfo(object):
    def __init__(self, decl): # [ funcname, return_ctype, [modifiers], [args] ]
        self.funcs = []
        self.funcs.append( FuncInfo(decl) )
        self.jname = self.funcs[0].jname
        self.isconstructor = self.funcs[0].name == self.funcs[0].classname



    def add_func(self, fi):
        self.funcs.append( fi )


class JavaWrapperGenerator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.classes = { "Mat" : ClassInfo([ 'class Mat', '', [], [] ]) }
        self.module = ""
        self.Module = ""
        self.java_code= {} # { class : {j_code, jn_code} }
        self.cpp_code = None
        self.ported_func_list = []
        self.skipped_func_list = []
        self.def_args_hist = {} # { def_args_cnt : funcs_cnt }
        self.classes_map = []
        self.classes_simple = []

    def add_class_code_stream(self, class_name, cls_base = ''):
        jname = self.classes[class_name].jname
        self.java_code[class_name] = { "j_code" : StringIO(), "jn_code" : StringIO(), }
        if class_name != self.Module:
            if cls_base:
                self.java_code[class_name]["j_code"].write("""
//
// This file is auto-generated. Please don't modify it!
//
package org.berkeleyvision.%(m)s;

$imports

// C++: class %(c)s
//javadoc: %(c)s
public class %(jc)s extends %(base)s {

    protected %(jc)s(long addr) { super(addr); }

""" % { 'm' : self.module, 'c' : class_name, 'jc' : jname, 'base' :  cls_base })
            else: # not cls_base
                self.java_code[class_name]["j_code"].write("""
//
// This file is auto-generated. Please don't modify it!
//
package org.berkeleyvision.%(m)s;

$imports

// C++: class %(c)s
//javadoc: %(c)s
public class %(jc)s {

    protected final long nativeObj;
    protected %(jc)s(long addr) { nativeObj = addr; }

""" % { 'm' : self.module, 'c' : class_name, 'jc' : jname })
        else: # class_name == self.Module
            self.java_code[class_name]["j_code"].write("""
//
// This file is auto-generated. Please don't modify it!
//
package org.berkeleyvision.%(m)s;

$imports

public class %(jc)s {
""" % { 'm' : self.module, 'jc' : jname } )

        if class_name == 'Core':
            (major, minor, revision, status) = getLibVersion(
                (os.path.dirname(__file__) or '.') + '/../../core/include/opencv2/core/version.hpp')
            version_str    = '.'.join( (major, minor, revision) ) + status
            version_suffix =  ''.join( (major, minor, revision) )
            self.classes[class_name].imports.add("java.lang.String")
            self.java_code[class_name]["j_code"].write("""
    // these constants are wrapped inside functions to prevent inlining
    private static String getVersion() { return "%(v)s"; }
    private static String getNativeLibraryName() { return "opencv_java%(vs)s"; }
    private static int getVersionMajor() { return %(ma)s; }
    private static int getVersionMinor() { return %(mi)s; }
    private static int getVersionRevision() { return %(re)s; }
    private static String getVersionStatus() { return "%(st)s"; }

    public static final String VERSION = getVersion();
    public static final String NATIVE_LIBRARY_NAME = getNativeLibraryName();
    public static final int VERSION_MAJOR = getVersionMajor();
    public static final int VERSION_MINOR = getVersionMinor();
    public static final int VERSION_REVISION = getVersionRevision();
    public static final String VERSION_STATUS = getVersionStatus();
""" % { 'v' : version_str, 'vs' : version_suffix, 'ma' : major, 'mi' : minor, 're' : revision, 'st': status } )


    def add_class(self, decl):
        classinfo = ClassInfo(decl)
        if classinfo.name in class_ignore_list:
            return
        name = classinfo.name
        if name in self.classes:
            print "Generator error: class %s (%s) is duplicated" % \
                    (name, classinfo.cname)
            return
        self.classes[name] = classinfo
        if name in type_dict:
            print "Duplicated class: " + name
            return
        if '/Simple' in decl[2]:
            self.classes_simple.append(name)
        if ('/Map' in decl[2]):
            self.classes_map.append(name)
            #adding default c-tor
            ffi = FuncFamilyInfo(['cv.'+name+'.'+name, '', [], []])
            classinfo.methods[ffi.jname] = ffi
        type_dict[name] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "(*("+name+"*)%(n)s_nativeObj)", "jni_type" : "jlong",
              "suffix" : "J" }
        type_dict[name+'*'] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "("+name+"*)%(n)s_nativeObj", "jni_type" : "jlong",
              "suffix" : "J" }

        # missing_consts { Module : { public : [[name, val],...], private : [[]...] } }
        if name in missing_consts:
            if 'private' in missing_consts[name]:
                for (n, val) in missing_consts[name]['private']:
                    classinfo.private_consts.append( ConstInfo(n, n, val, True) )
            if 'public' in missing_consts[name]:
                for (n, val) in missing_consts[name]['public']:
                    classinfo.consts.append( ConstInfo(n, n, val, True) )

        # class props
        for p in decl[3]:
            if True: #"vector" not in p[0]:
                classinfo.props.append( ClassPropInfo(p) )
            else:
                print "Skipped property: [%s]" % name, p

        self.add_class_code_stream(name, classinfo.base)
        if classinfo.base:
            self.get_imports(name, classinfo.base)


    def add_const(self, decl): # [ "const cname", val, [], [] ]
        name = decl[0].replace("const ", "").strip()
        name = re.sub(r"^cv\.", "", name)
        cname = name.replace(".", "::")
        for c in const_ignore_list:
            if re.match(c, name):
                return
        # class member?
        dpos = name.rfind(".")
        if dpos >= 0:
            classname = name[:dpos]
            name = name[dpos+1:]
        else:
            classname = self.Module
        if classname not in self.classes:
            # this class isn't wrapped
            # skipping this const
            return

        consts = self.classes[classname].consts
        for c in const_private_list:
            if re.match(c, name):
                consts = self.classes[classname].private_consts
                break

        constinfo = ConstInfo(cname, name, decl[1])
        # checking duplication
        for list in self.classes[classname].consts, self.classes[classname].private_consts:
            for c in list:
                if c.name == constinfo.name:
                    if c.addedManually:
                        return
                    print "Generator error: constant %s (%s) is duplicated" \
                            % (constinfo.name, constinfo.cname)
                    sys.exit(-1)

        consts.append(constinfo)

    def add_func(self, decl):
        ffi = FuncFamilyInfo(decl)
        classname = ffi.funcs[0].classname or self.Module
        if classname in class_ignore_list:
            return
        if classname in ManualFuncs and ffi.jname in ManualFuncs[classname]:
            return
        if classname not in self.classes:
            print "Generator error: the class %s for method %s is missing" % \
                    (classname, ffi.jname)
            sys.exit(-1)
        func_map = self.classes[classname].methods
        if ffi.jname in func_map:
            func_map[ffi.jname].add_func(ffi.funcs[0])
        else:
            func_map[ffi.jname] = ffi
        # calc args with def val
        cnt = len([a for a in ffi.funcs[0].args if a.defval])
        self.def_args_hist[cnt] = self.def_args_hist.get(cnt, 0) + 1

    def save(self, path, buf):
        f = open(path, "wt")
        f.write(buf)
        f.close()

    def gen(self, srcfiles, module, output_path):
        self.clear()
        self.module = module
        self.Module = module.capitalize()
        parser = hdr_parser.CppHeaderParser()

        self.add_class( ['class ' + self.Module, '', [], []] ) # [ 'class/struct cname', ':bases', [modlist] [props] ]

        # scan the headers and build more descriptive maps of classes, consts, functions
        for hdr in srcfiles:
            decls = parser.parse(hdr)
            for decl in decls:
                name = decl[0]
                if name.startswith("struct") or name.startswith("class"):
                    self.add_class(decl)
                elif name.startswith("const"):
                    self.add_const(decl)
                else: # function
                    self.add_func(decl)

        self.cpp_code = StringIO()
        self.cpp_code.write(Template("""
//
// This file is auto-generated, please don't edit!
//

#define LOG_TAG "org.berkeleyvision.$m"

#include "common.h"

#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_$M

#include <string>

#include "opencv2/$m.hpp"

using namespace cv;

/// throw java exception
static void throwJavaException(JNIEnv *env, const std::exception *e, const char *method) {
  std::string what = "unknown exception";
  jclass je = 0;

  if(e) {
    std::string exception_type = "std::exception";

    if(dynamic_cast<const cv::Exception*>(e)) {
      exception_type = "cv::Exception";
      je = env->FindClass("org/opencv/core/CvException");
    }

    what = exception_type + ": " + e->what();
  }

  if(!je) je = env->FindClass("java/lang/Exception");
  env->ThrowNew(je, what.c_str());

  LOGE("%s caught %s", method, what.c_str());
  (void)method;        // avoid "unused" warning
}


extern "C" {

""").substitute( m = module, M = module.upper() ) )

        # generate code for the classes
        for name in self.classes.keys():
            if name == "Mat":
                continue
            self.gen_class(name)
            # saving code streams
            imports = "\n".join([ "import %s;" % c for c in \
                sorted(self.classes[name].imports) if not c.startswith('org.berkeleyvision.'+self.module) ])
            self.java_code[name]["j_code"].write("\n\n%s\n}\n" % self.java_code[name]["jn_code"].getvalue())
            java_code = self.java_code[name]["j_code"].getvalue()
            java_code = Template(java_code).substitute(imports = imports)
            self.save("%s/%s+%s.java" % (output_path, module, self.classes[name].jname), java_code)

        self.cpp_code.write( '\n} // extern "C"\n\n#endif // HAVE_OPENCV_%s\n' % module.upper() )
        self.save(output_path+"/"+module+".cpp",  self.cpp_code.getvalue())

        # report
        report = StringIO()
        report.write("PORTED FUNCs LIST (%i of %i):\n\n" % \
            (len(self.ported_func_list), len(self.ported_func_list)+ len(self.skipped_func_list))
        )
        report.write("\n".join(self.ported_func_list))
        report.write("\n\nSKIPPED FUNCs LIST (%i of %i):\n\n" % \
            (len(self.skipped_func_list), len(self.ported_func_list)+ len(self.skipped_func_list))
        )
        report.write("".join(self.skipped_func_list))

        for i in self.def_args_hist.keys():
            report.write("\n%i def args - %i funcs" % (i, self.def_args_hist[i]))

        report.write("\n\nclass as MAP:\n\t" + "\n\t".join(self.classes_map))
        report.write("\n\nclass SIMPLE:\n\t" + "\n\t".join(self.classes_simple))

        self.save(output_path+"/"+module+".txt", report.getvalue())

        #print "Done %i of %i funcs." % (len(self.ported_func_list), len(self.ported_func_list)+ len(self.skipped_func_list))



    def get_imports(self, scope_classname, ctype):
        imports = self.classes[scope_classname or self.Module].imports
        if ctype.startswith('vector_vector'):
            imports.add("org.berkeleyvision.core.Mat")
            imports.add("java.util.List")
            imports.add("org.berkeleyvision.utils.Converters")
            self.get_imports(scope_classname, ctype.replace('vector_vector', 'vector'))
            return
        if ctype.startswith('vector'):
            imports.add("org.berkeleyvision.core.Mat")
            if type_dict[ctype]['j_type'].startswith('MatOf'):
                imports.add("org.berkeleyvision.core." + type_dict[ctype]['j_type'])
                return
            else:
                imports.add("java.util.List")
                imports.add("org.berkeleyvision.utils.Converters")
                self.get_imports(scope_classname, ctype.replace('vector_', ''))
                return
        j_type = ''
        if ctype in type_dict:
            j_type = type_dict[ctype]['j_type']
        elif ctype in ("Algorithm"):
            j_type = ctype
        if j_type in ( "CvType", "Mat", "Point", "Point3", "Range", "Rect", "RotatedRect", "Scalar", "Size", "TermCriteria", "Algorithm" ):
            imports.add("org.berkeleyvision.core." + j_type)
        if j_type == 'String':
            imports.add("java.lang.String")
        return



    def gen_func(self, fi, prop_name=''):
        j_code   = self.java_code[fi.classname or self.Module]["j_code"]
        jn_code  = self.java_code[fi.classname or self.Module]["jn_code"]
        cpp_code = self.cpp_code

        # c_decl
        # e.g: void add(Mat src1, Mat src2, Mat dst, Mat mask = Mat(), int dtype = -1)
        if prop_name:
            c_decl = "%s %s::%s" % (fi.ctype, fi.classname, prop_name)
        else:
            decl_args = []
            for a in fi.args:
                s = a.ctype or ' _hidden_ '
                if a.pointer:
                    s += "*"
                elif a.out:
                    s += "&"
                s += " " + a.name
                if a.defval:
                    s += " = "+a.defval
                decl_args.append(s)
            c_decl = "%s %s %s(%s)" % ( fi.static, fi.ctype, fi.cname, ", ".join(decl_args) )

        # java comment
        j_code.write( "\n    //\n    // C++: %s\n    //\n\n" % c_decl )
        # check if we 'know' all the types
        if fi.ctype not in type_dict: # unsupported ret type
            msg = "// Return type '%s' is not supported, skipping the function\n\n" % fi.ctype
            self.skipped_func_list.append(c_decl + "\n" + msg)
            j_code.write( " "*4 + msg )
            print "SKIP:", c_decl.strip(), "\t due to RET type", fi.ctype
            return
        for a in fi.args:
            if a.ctype not in type_dict:
                if not a.defval and a.ctype.endswith("*"):
                    a.defval = 0
                if a.defval:
                    a.ctype = ''
                    continue
                msg = "// Unknown type '%s' (%s), skipping the function\n\n" % (a.ctype, a.out or "I")
                self.skipped_func_list.append(c_decl + "\n" + msg)
                j_code.write( " "*4 + msg )
                print "SKIP:", c_decl.strip(), "\t due to ARG type", a.ctype, "/" + (a.out or "I")
                return

        self.ported_func_list.append(c_decl)

        # jn & cpp comment
        jn_code.write( "\n    // C++: %s\n" % c_decl )
        cpp_code.write( "\n//\n// %s\n//\n" % c_decl )

        # java args
        args = fi.args[:] # copy
        suffix_counter = int( self.classes[fi.classname or self.Module].methods_suffixes.get(fi.jname, -1) )
        while True:
            suffix_counter += 1
            self.classes[fi.classname or self.Module].methods_suffixes[fi.jname] = suffix_counter
             # java native method args
            jn_args = []
            # jni (cpp) function args
            jni_args = [ArgInfo([ "env", "env", "", [], "" ]), ArgInfo([ "cls", "", "", [], "" ])]
            j_prologue = []
            j_epilogue = []
            c_prologue = []
            c_epilogue = []
            if type_dict[fi.ctype]["jni_type"] == "jdoubleArray":
                fields = type_dict[fi.ctype]["jn_args"]
                c_epilogue.append( \
                    ("jdoubleArray _da_retval_ = env->NewDoubleArray(%(cnt)i);  " +
                     "jdouble _tmp_retval_[%(cnt)i] = {%(args)s}; " +
                     "env->SetDoubleArrayRegion(_da_retval_, 0, %(cnt)i, _tmp_retval_);") %
                    { "cnt" : len(fields), "args" : ", ".join(["_retval_" + f[1] for f in fields]) } )
            if fi.classname and fi.ctype and not fi.static: # non-static class method except c-tor
                # adding 'self'
                jn_args.append ( ArgInfo([ "__int64", "nativeObj", "", [], "" ]) )
                jni_args.append( ArgInfo([ "__int64", "self", "", [], "" ]) )
            self.get_imports(fi.classname, fi.ctype)
            for a in args:
                if not a.ctype: # hidden
                    continue
                self.get_imports(fi.classname, a.ctype)
                if "vector" in a.ctype: # pass as Mat
                    jn_args.append  ( ArgInfo([ "__int64", "%s_mat.nativeObj" % a.name, "", [], "" ]) )
                    jni_args.append ( ArgInfo([ "__int64", "%s_mat_nativeObj" % a.name, "", [], "" ]) )
                    c_prologue.append( type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";" )
                    c_prologue.append( "Mat& %(n)s_mat = *((Mat*)%(n)s_mat_nativeObj)" % {"n" : a.name} + ";" )
                    if "I" in a.out or not a.out:
                        if a.ctype.startswith("vector_vector_"):
                            self.classes[fi.classname or self.Module].imports.add("java.util.ArrayList")
                            j_prologue.append( "List<Mat> %(n)s_tmplm = new ArrayList<Mat>((%(n)s != null) ? %(n)s.size() : 0);" % {"n" : a.name } )
                            j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s, %(n)s_tmplm);" % {"n" : a.name, "t" : a.ctype} )
                        else:
                            if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                                j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s);" % {"n" : a.name, "t" : a.ctype} )
                            else:
                                j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                        c_prologue.append( "Mat_to_%(t)s( %(n)s_mat, %(n)s );" % {"n" : a.name, "t" : a.ctype} )
                    else:
                        if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                            j_prologue.append( "Mat %s_mat = new Mat();" % a.name )
                        else:
                            j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                    if "O" in a.out:
                        if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                            j_epilogue.append("Converters.Mat_to_%(t)s(%(n)s_mat, %(n)s);" % {"t" : a.ctype, "n" : a.name})
                        c_epilogue.append( "%(t)s_to_Mat( %(n)s, %(n)s_mat );" % {"n" : a.name, "t" : a.ctype} )
                else:
                    fields = type_dict[a.ctype].get("jn_args", ((a.ctype, ""),))
                    if "I" in a.out or not a.out or a.ctype in self.classes: # input arg, pass by primitive fields
                        for f in fields:
                            jn_args.append ( ArgInfo([ f[0], a.name + f[1], "", [], "" ]) )
                            jni_args.append( ArgInfo([ f[0], a.name + f[1].replace(".","_").replace("[","").replace("]",""), "", [], "" ]) )
                    if a.out and a.ctype not in self.classes: # out arg, pass as double[]
                        jn_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        j_prologue.append( "double[] %s_out = new double[%i];" % (a.name, len(fields)) )
                        c_epilogue.append( \
                            "jdouble tmp_%(n)s[%(cnt)i] = {%(args)s}; env->SetDoubleArrayRegion(%(n)s_out, 0, %(cnt)i, tmp_%(n)s);" %
                            { "n" : a.name, "cnt" : len(fields), "args" : ", ".join([a.name + f[1] for f in fields]) } )
                        if a.ctype in ('bool', 'int', 'long', 'float', 'double'):
                            j_epilogue.append('if(%(n)s!=null) %(n)s[0] = (%(t)s)%(n)s_out[0];' % {'n':a.name,'t':a.ctype})
                        else:
                            set_vals = []
                            i = 0
                            for f in fields:
                                set_vals.append( "%(n)s%(f)s = %(t)s%(n)s_out[%(i)i]" %
                                    {"n" : a.name, "t": ("("+type_dict[f[0]]["j_type"]+")", "")[f[0]=="double"], "f" : f[1], "i" : i}
                                )
                                i += 1
                            j_epilogue.append( "if("+a.name+"!=null){ " + "; ".join(set_vals) + "; } ")


            # java part:
            # private java NATIVE method decl
            # e.g.
            # private static native void add_0(long src1, long src2, long dst, long mask, int dtype);
            jn_code.write( Template(\
                "    private static native $type $name($args);\n").substitute(\
                type = type_dict[fi.ctype].get("jn_type", "double[]"), \
                name = fi.jname + '_' + str(suffix_counter), \
                args = ", ".join(["%s %s" % (type_dict[a.ctype]["jn_type"], a.name.replace(".","_").replace("[","").replace("]","")) for a in jn_args])
            ) );

            # java part:

            #java doc comment
            f_name = fi.name
            if fi.classname:
                f_name = fi.classname + "::" + fi.name
            java_doc = "//javadoc: " + f_name + "(%s)" % ", ".join([a.name for a in args if a.ctype])
            j_code.write(" "*4 + java_doc + "\n")

            # public java wrapper method impl (calling native one above)
            # e.g.
            # public static void add( Mat src1, Mat src2, Mat dst, Mat mask, int dtype )
            # { add_0( src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj, dtype );  }
            ret_type = fi.ctype
            if fi.ctype.endswith('*'):
                ret_type = ret_type[:-1]
            ret_val = type_dict[ret_type]["j_type"] + " retVal = "
            tail = ""
            ret = "return retVal;"
            if ret_type.startswith('vector'):
                tail = ")"
                j_type = type_dict[ret_type]["j_type"]
                if j_type.startswith('MatOf'):
                    ret_val += j_type + ".fromNativeAddr("
                else:
                    ret_val = "Mat retValMat = new Mat("
                    j_prologue.append( j_type + ' retVal = new Array' + j_type+'();')
                    self.classes[fi.classname or self.Module].imports.add('java.util.ArrayList')
                    j_epilogue.append('Converters.Mat_to_' + ret_type + '(retValMat, retVal);')
            elif ret_type == "void":
                ret_val = ""
                ret = "return;"
            elif ret_type == "": # c-tor
                if fi.classname and self.classes[fi.classname].base:
                    ret_val = "super( "
                    tail = " )"
                else:
                    ret_val = "nativeObj = "
                ret = "return;"
            elif ret_type in self.classes: # wrapped class
                ret_val = type_dict[ret_type]["j_type"] + " retVal = new " + self.classes[ret_type].jname + "("
                tail = ")"
            elif "jn_type" not in type_dict[ret_type]:
                ret_val = type_dict[fi.ctype]["j_type"] + " retVal = new " + type_dict[ret_type]["j_type"] + "("
                tail = ")"

            static = "static"
            if fi.classname:
                static = fi.static

            j_args = []
            for a in args:
                if not a.ctype: #hidden
                    continue
                jt = type_dict[a.ctype]["j_type"]
                if a.out and a.ctype in ('bool', 'int', 'long', 'float', 'double'):
                    jt += '[]'
                j_args.append( jt + ' ' + a.name )

            j_code.write( Template(\
"""    public $static $j_type $j_name($j_args)
    {
        $prologue
        $ret_val$jn_name($jn_args_call)$tail;
        $epilogue
        $ret
    }

"""
                ).substitute(\
                    ret = ret, \
                    ret_val = ret_val, \
                    tail = tail, \
                    prologue = "\n        ".join(j_prologue), \
                    epilogue = "\n        ".join(j_epilogue), \
                    static=static, \
                    j_type=type_dict[fi.ctype]["j_type"], \
                    j_name=fi.jname, \
                    j_args=", ".join(j_args), \
                    jn_name=fi.jname + '_' + str(suffix_counter), \
                    jn_args_call=", ".join( [a.name for a in jn_args] ),\
                )
            )


            # cpp part:
            # jni_func(..) { _retval_ = cv_func(..); return _retval_; }
            ret = "return _retval_;"
            default = "return 0;"
            if fi.ctype == "void":
                ret = "return;"
                default = "return;"
            elif not fi.ctype: # c-tor
                ret = "return (jlong) _retval_;"
            elif fi.ctype.startswith('vector'): # c-tor
                ret = "return (jlong) _retval_;"
            elif fi.ctype == "String":
                ret = "return env->NewStringUTF(_retval_.c_str());"
                default = 'return env->NewStringUTF("");'
            elif fi.ctype in self.classes: # wrapped class:
                ret = "return (jlong) new %s(_retval_);" % fi.ctype
            elif ret_type in self.classes: # pointer to wrapped class:
                ret = "return (jlong) _retval_;"
            elif type_dict[fi.ctype]["jni_type"] == "jdoubleArray":
                ret = "return _da_retval_;"

            # hack: replacing func call with property set/get
            name = fi.name
            if prop_name:
                if args:
                    name = prop_name + " = "
                else:
                    name = prop_name + ";//"

            cvname = "cv::" + name
            retval = fi.ctype + " _retval_ = "
            if fi.ctype == "void":
                retval = ""
            elif fi.ctype == "String":
                retval = "cv::" + retval
            elif fi.ctype.startswith('vector'):
                retval = type_dict[fi.ctype]['jni_var'] % {"n" : '_ret_val_vector_'} + " = "
                c_epilogue.append("Mat* _retval_ = new Mat();")
                c_epilogue.append(fi.ctype+"_to_Mat(_ret_val_vector_, *_retval_);")
            if fi.classname:
                if not fi.ctype: # c-tor
                    retval = fi.classname + "* _retval_ = "
                    cvname = "new " + fi.classname
                elif fi.static:
                    cvname = "%s::%s" % (fi.classname, name)
                else:
                    cvname = "me->" + name
                    c_prologue.append(\
                        "%(cls)s* me = (%(cls)s*) self; //TODO: check for NULL" \
                            % { "cls" : fi.classname} \
                    )
            cvargs = []
            for a in args:
                if a.pointer:
                    jni_name = "&%(n)s"
                else:
                    jni_name = "%(n)s"
                    if not a.out and not "jni_var" in type_dict[a.ctype]:
                        # explicit cast to C type to avoid ambiguous call error on platforms (mingw)
                        # where jni types are different from native types (e.g. jint is not the same as int)
                        jni_name  = "(%s)%s" % (a.ctype, jni_name)
                if not a.ctype: # hidden
                    jni_name = a.defval
                cvargs.append( type_dict[a.ctype].get("jni_name", jni_name) % {"n" : a.name})
                if "vector" not in a.ctype :
                    if ("I" in a.out or not a.out or a.ctype in self.classes) and "jni_var" in type_dict[a.ctype]: # complex type
                        c_prologue.append(type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";")
                    if a.out and "I" not in a.out and a.ctype not in self.classes and a.ctype:
                        c_prologue.append("%s %s;" % (a.ctype, a.name))

            rtype = type_dict[fi.ctype].get("jni_type", "jdoubleArray")
            clazz = self.Module
            if fi.classname:
                clazz = self.classes[fi.classname].jname
            cpp_code.write ( Template( \
"""
JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname ($argst);

JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname
  ($args)
{
    static const char method_name[] = "$module::$fname()";
    try {
        LOGD("%s", method_name);
        $prologue
        $retval$cvname( $cvargs );
        $epilogue$ret
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    $default
}


""" ).substitute( \
        rtype = rtype, \
        module = self.module, \
        clazz = clazz.replace('_', '_1'), \
        fname = (fi.jname + '_' + str(suffix_counter)).replace('_', '_1'), \
        args  = ", ".join(["%s %s" % (type_dict[a.ctype].get("jni_type"), a.name) for a in jni_args]), \
        argst = ", ".join([type_dict[a.ctype].get("jni_type") for a in jni_args]), \
        prologue = "\n        ".join(c_prologue), \
        epilogue = "  ".join(c_epilogue) + ("\n        " if c_epilogue else ""), \
        ret = ret, \
        cvname = cvname, \
        cvargs = ", ".join(cvargs), \
        default = default, \
        retval = retval, \
    ) )

            # processing args with default values
            if not args or not args[-1].defval:
                break
            while args and args[-1].defval:
                # 'smart' overloads filtering
                a = args.pop()
                if a.name in ('mask', 'dtype', 'ddepth', 'lineType', 'borderType', 'borderMode', 'criteria'):
                    break



    def gen_class(self, name):
        # generate code for the class
        ci = self.classes[name]
        # constants
        if ci.private_consts:
            self.java_code[name]['j_code'].write("""
    private static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in ci.private_consts])
            )
        if ci.consts:
            self.java_code[name]['j_code'].write("""
    public static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in ci.consts])
            )
        # c-tors
        fflist = ci.methods.items()
        fflist.sort()
        for n, ffi in fflist:
            if ffi.isconstructor:
                for fi in ffi.funcs:
                    fi.jname = ci.jname
                    self.gen_func(fi)
        # other methods
        for n, ffi in fflist:
            if not ffi.isconstructor:
                for fi in ffi.funcs:
                    self.gen_func(fi)
        # props
        for pi in ci.props:
            # getter
            getter_name = name + ".get_" + pi.name
            #print getter_name
            fi = FuncInfo( [getter_name, pi.ctype, [], []] ) # [ funcname, return_ctype, [modifiers], [args] ]
            self.gen_func(fi, pi.name)
            if pi.rw:
                #setter
                setter_name = name + ".set_" + pi.name
                #print setter_name
                fi = FuncInfo( [ setter_name, "void", [], [ [pi.ctype, pi.name, "", [], ""] ] ] )
                self.gen_func(fi, pi.name)

        # manual ports
        if name in ManualFuncs:
            for func in ManualFuncs[name].keys():
                self.java_code[name]["j_code"].write ( ManualFuncs[name][func]["j_code"] )
                self.java_code[name]["jn_code"].write( ManualFuncs[name][func]["jn_code"] )
                self.cpp_code.write( ManualFuncs[name][func]["cpp_code"] )

        if name != self.Module:
            # finalize()
            self.java_code[name]["j_code"].write(
"""
    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }
""" )

            self.java_code[name]["jn_code"].write(
"""
    // native support for java finalize()
    private static native void delete(long nativeObj);
""" )

            # native support for java finalize()
            self.cpp_code.write( \
"""
//
//  native support for java finalize()
//  static void %(cls)s::delete( __int64 self )
//
JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete(JNIEnv*, jclass, jlong);

JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete
  (JNIEnv*, jclass, jlong self)
{
    delete (%(cls)s*) self;
}

""" % {"module" : module, "cls" : name, "j_cls" : ci.jname.replace('_', '_1')}
            )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage:\n", \
            os.path.basename(sys.argv[0]), \
            "<full path to hdr_parser.py> <module name> <C++ header> [<C++ header>...]"
        print "Current args are: ", ", ".join(["'"+a+"'" for a in sys.argv])
        exit(0)

    dstdir = "."
    hdr_parser_path = os.path.abspath(sys.argv[1])
    if hdr_parser_path.endswith(".py"):
        hdr_parser_path = os.path.dirname(hdr_parser_path)
    sys.path.append(hdr_parser_path)
    import hdr_parser
    module = sys.argv[2]
    srcfiles = sys.argv[3:]
    #print "Generating module '" + module + "' from headers:\n\t" + "\n\t".join(srcfiles)
    generator = JavaWrapperGenerator()
    generator.gen(srcfiles, module, dstdir)
