#!/usr/bin/env python
# Copyright (c) 2012 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Determine OS and various other system properties.

Determine the name of the platform used and other system properties such as
the location of Chrome.  This is used, for example, to determine the correct
Toolchain to invoke.
"""

import optparse
import os
import re
import subprocess
import sys

##import oshelpers


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if sys.version_info < (2, 6, 0):
  sys.stderr.write("python 2.6 or later is required run this script\n")
  sys.exit(1)


class Error(Exception):
  pass

def GetPlatform():
  if sys.platform.startswith('cygwin') or sys.platform.startswith('win'):
    return 'win'
  elif sys.platform.startswith('darwin'):
    return 'mac'
  elif sys.platform.startswith('linux'):
    return 'linux'
  else:
    raise Error("Unknown platform: %s" % sys.platform)


## The above is part of
## http://src.chromium.org/chrome/trunk/src/native_client_sdk/src/tools/getos.py
  
"""Determine OS distributor and install caffe's dependencies

Determine OS distributor to choose the proper package management system.
This script currently only supports Ubuntu.
TODO: RHEL/Centos/Fedora, Mac etc.
"""
def get_linux_distributor(platform):
  platform = platform.lower()
  if platform == 'win':
    return 'microsoft'
    
  if platform == 'mac':
    return 'apple'

  distributor = None
  if platform == 'linux':
    try:
      pobj = subprocess.Popen(['lsb_release', '-i'], stdout= subprocess.PIPE)
      distributor = pobj.communicate()[0]
      distributor = distributor.split(':')[-1].strip().lower()
      if distributor.startswith('ubuntu'):
        distributor = 'ubuntu'
    except Exception:
      pass
  return distributor

def get_distribution_version(platform):
  version = None
  if platform == 'linux':
    try:
      pobj = subprocess.Popen(['lsb_release', '-r'], stdout= subprocess.PIPE)
      version = pobj.communicate()[0]
      version = version.split(':')[-1].strip().lower()
    except Exception:
      pass
  return version


def install_caffe_dependencies():
  platform = GetPlatform()
  distributor = get_linux_distributor(platform)
  dist_version = get_distribution_version(platform)
  dev_libs = ['protobuf', 'leveldb', 'snappy', 'opencv', 'atlas-base']
  ubuntu_boost_version = {'13.10':'1.53', '12.10':'1.50',
                          '12.04':'1.48', '12.04.3':'1.48'}
  if distributor == 'ubuntu':
    boost_version = ubuntu_boost_version[dist_version]
    dev_libs.append('boost' + boost_version)
    cmd = 'sudo apt-get -y --force-yes install ' + \
          ' '.join(['lib%s-dev' % lib for lib in dev_libs])
    os.system(cmd)
    try:
      print cmd
      subprocess.Popen(cmd, stdout= subprocess.PIPE)
    except Exception:
      return Error('Failed to install dependencies(platform: %s, \
distributor: %s, release: %s)'
                   % (platform, distributor, dist_version))
    return 0


if __name__ == '__main__':
  try:
    sys.exit(install_caffe_dependencies())
  except Error as e:
    sys.stderr.write(str(e) + '\n')
    sys.exit(1)
