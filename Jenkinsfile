pipeline {

    // pick any available agent
    agent any

    parameters {

    }

    environment {
        PATH = "${PATH}:/usr/local/bin:/usr/bin"
        PYTHONPATH = "${WORKSPACE}/src"
        MLFLOW_TRACKING_URL="http://192.168.1.102:8000"
    }

    stages {
        stage('checkout the code from repo') {
            steps {
                checkout scm
            }
        }
    }

    stage('Setup Environment') {
        steps {
            script {
                sh '''
                    pip3 install -r requirements.txt --break-system-packages
                    pip3 install -r requirements-dev.txt --break-system-packages
                '''
            }
        }
    }
}