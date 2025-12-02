pipeline {

    // pick any available agent
    agent any

    // parameters {
    //     choice(name: 'ENVIRONMENT', choices: ['dev', 'staging', 'prod'], description: 'Deployment environment')
    //     booleanParam(name: 'RETRAIN', defaultValue: false, description: 'Retrain the model')
    //     string(name: 'MODEL_VERSION', defaultValue: '1.0.0', description: 'Model version')
    // }

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