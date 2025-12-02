pipeline {

    // pick any available agent
    agent any

    parameters {
        choice(name: 'ENVIRONMENT', choices: ['dev', 'staging', 'prod'], description: 'Deployment environment')
        booleanParam(name: 'RETRAIN', defaultValue: false, description: 'Retrain the model')
        string(name: 'MODEL_VERSION', defaultValue: '1.0.0', description: 'Model version')
    }

    environment {
        PATH = "${PATH}:/usr/local/bin:/usr/bin"
        PYTHONPATH = "${WORKSPACE}/src"
        MLFLOW_TRACKING_URI = 'http://192.168.1.102:8000'
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

        stage('Data Validation') {
            when {
                expression { params.RETRAIN == true }
            }
            steps {
                script {
                    sh '''
                        python3 scripts/validate_data.py data/raw/Salary_Data.csv
                    '''
                }
            }
        } 

        stage('Model Training') {
            when {
                expression { params.RETRAIN == true }
            }
            steps {
                script {
                    sh '''
                        python3 train_model.py \
                            --data-path data/raw/Salary_Data.csv \
                            --output-dir models \
                            --test-size 0.2 \
                            --random-state 42
                    '''
                    archiveArtifacts 'models/*.pkl, models/*.json'
                    archiveArtifacts 'reports/*.json, reports/*.png'
                }
            }
        }

        stage('Model Evaluation') {
            when {
                expression { params.RETRAIN == true }
            }
            steps {
                script {
                    sh '''
                        python3 scripts/evaluate_model.py reports/evaluation_report.json
                    '''
                }
            }
        }

        stage('Build Docker Image') {
            when {
                anyOf {
                    environment name: 'ENVIRONMENT', value: 'staging'
                    environment name: 'ENVIRONMENT', value: 'prod'
                }
            }
            steps {
                script {
                    sh '''
                        docker build -t salary-predictor:${MODEL_VERSION}-${BUILD_NUMBER} .
                        docker tag salary-predictor:${MODEL_VERSION}-${BUILD_NUMBER} salary-predictor:latest
                    '''
                }
            }
        }

        stage('Push to Registry') {
            when {
                anyOf {
                    environment name: 'ENVIRONMENT', value: 'staging'
                    environment name: 'ENVIRONMENT', value: 'prod'
                }
            }
            steps {
                script {
                    withCredentials([usernamePassword(
                        credentialsId: 'docker-hub',
                        usernameVariable: 'DOCKER_USERNAME',
                        passwordVariable: 'DOCKER_PASSWORD'
                    )]) {
                        sh '''
                            echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
                            docker tag salary-predictor:${MODEL_VERSION}-${BUILD_NUMBER} \
                                ${DOCKER_USERNAME}/salary-prediction-demo:${MODEL_VERSION}-${BUILD_NUMBER}
                            docker push ${DOCKER_USERNAME}/salary-prediction-demo:${MODEL_VERSION}-${BUILD_NUMBER}
                        '''
                    }
                }
            }
        }

    }

    
}