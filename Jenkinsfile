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

        stage('Code Quality') {
            steps {
                script {
                    sh '''
                        # echo "Running Black..."
                        # black --check src/ tests/
                        
                        # echo "Running Flake8..."
                        # flake8 src/ tests/
                        
                        # echo "Running MyPy..."
                        # mypy src/
                        
                        echo "Running Pylint..."
                        # pylint src/
                    '''
                }
            }
        }

        stage('Unit Tests') {
            steps {
                script {
                    sh '''                                             
                        export PYTHONPATH="${WORKSPACE}/src:${PYTHONPATH}"
                        echo "PYTHONPATH: ${PYTHONPATH}"
                        
                        # Install package in development mode
                        pip3 install -e . --break-system-packages || echo "Install in dev mode failed, continuing..."
                        
                        # Run tests with coverage for the src directory
                        python3 -m pytest tests/ \
                            -v \
                            --cov=src \
                            --cov-report=xml:coverage.xml \
                            --cov-report=html:htmlcov \
                            --cov-report=term-missing \
                            --junitxml=test-results.xml
                    '''

                    junit 'test-results.xml'
                    publishHTML(target: [
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
                
            }
        }
        

    }

    
}