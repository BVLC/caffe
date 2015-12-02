<!--
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
-->

# Soporte shell Bash

Cordova CLI viene junto con un script que proporciona ficha-terminación de línea de comandos para Bash. Si usted está utilizando un sistema operativo bastante Unix-y (Linux, BSD, OS X) usted puede instalar esto para hacer más fácil escribir líneas de comandos cordova.

## Instalación

### Linux

Para instalar en un sistema Linux o BSD, copiar el `scripts/cordova.completion` de archivos a tu directorio `/etc/bash_completion.d`. Esto se leerá la próxima vez que empieces una nueva shell (línea de comandos).

### OS X

En OS X, ponga el `scripts/cordova.completion` archivo legible en cualquier lugar y añadir la siguiente línea al final de su archivo `~/.bashrc` :

    source <path to>/cordova.completion
    

Esto se leerá la próxima vez que empieces una nueva shell.

## Uso

¡Es Fácil! Siempre y cuando la linea comienze con un ejecutable llamado 'cordova', sólo presiona `<TAB>` en cualquier momento para ver la lista de terminaciones vólidas.

Ejemplos:

    $ cordova <TAB>
    build     compile   create    emulate   platform  plugin    prepare   serve
    
    $ cordova pla<TAB>
    
    $ cordova platform <TAB>
    add ls remove rm
    
    $ cordova platform a<TAB>
    
    $ cordova platform add <TAB>
    android     blackberry  ios         wp8         www
    
    $ cordova plugin rm <TAB>
    
    $ cordova plugin rm org.apache.cordova.<TAB>
    org.apache.cordova.file    org.apache.cordova.inappbrowser