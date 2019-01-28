pipeline {
  agent any
  stages {
    stage('Build GPU') {
      steps {
        sh '''ln -s Makefile.config.gpu.cudnn Makefile.config
make -j12'''
      }
    }
  }
}