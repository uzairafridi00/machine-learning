## Deploying Web Applications on AWS  

Amazon Web Services (AWS) provides a scalable and reliable cloud computing platform for hosting web applications. 

Below are the key steps to deploy a web application on AWS:  

### 1. **Set Up an AWS Account**  
   - Sign up at [AWS Console](https://aws.amazon.com/console/)  
   - Configure billing and security settings  

### 2. **Choose a Deployment Service**  
   Depending on your application type, select a suitable AWS service:  
   - **Amazon EC2** – Virtual servers for full control  
   - **AWS Elastic Beanstalk** – Automated deployment and scaling  
   - **AWS Lambda** – Serverless execution for microservices  
   - **Amazon Lightsail** – Simplified hosting for small applications  

### 3. **Prepare the Application**  
   - Ensure all dependencies are installed  
   - Configure environment variables  
   - Optimize the application for cloud deployment  

### 4. **Set Up the Infrastructure**  
   - **Networking:** Configure a Virtual Private Cloud (VPC)  
   - **Storage:** Use Amazon S3 for static assets  
   - **Database:** Choose from RDS, DynamoDB, or Aurora  

### 5. **Deploy the Application**  
   - **Manual Deployment:** Upload code via AWS CLI or EC2  
   - **Automated Deployment:** Use AWS CodeDeploy, CI/CD pipelines (CodePipeline, GitHub Actions)  

### 6. **Configure Domain & SSL**  
   - Register a domain via Route 53  
   - Set up HTTPS using AWS Certificate Manager and CloudFront  

### 7. **Monitor and Scale**  
   - **Logging & Monitoring:** Use CloudWatch for performance insights  
   - **Auto Scaling:** Configure Load Balancer and Auto Scaling Groups  
   - **Security:** Use IAM roles, security groups, and AWS WAF  

### 8. **Optimize & Maintain**  
   - Regularly update dependencies and patches  
   - Optimize costs using AWS Pricing Calculator  
   - Backup data with AWS Backup  


____

## Step by Step Guide to Deploy Web-app on AWS EC2 Instance

1. **Create an AWS account**
- Go to AWS and create an account.
- Sign in to the AWS Management Console.
- Open the Amazon EC2 console at https://console.aws.amazon.com/ec2/.

2. **Launch an EC2 instance**
- Create an EC2 instance by clicking on the Launch Instance button.
- Choose an Amazon Machine Image (AMI) - Ubuntu Server 20.04 LTS (HVM), SSD Volume Type.
- Choose an Instance Type - t2.micro (Free tier eligible) it may change based on your requirements.
- Configure Instance Details - Keep the default settings.
- Creata and save keypair for ssh access and do not share with anyone.
- Configure Security Group - Create a new security group and add rules to allow
   - HTTP and SSH traffic.
   - Pay attendtion to create 8501 port for flask and streamlit apps.
   - Add rule to access from anywhere.

- Add Storage - Keep the default settings.
- Add Tags - Keep the default settings.
- Review and Launch - Review the settings and click on the Launch button.

3. **Connect to your instance**
- Go to the EC2 dashboard and click on the Running Instances.
- Select the instance you just created and click on the Connect button.
- Follow the instructions to connect to your instance using SSH.
- Use the following command to connect to your instance:

```python
  chmod 400 your-key-pair.pem
  ssh -i "your-key-pair.pem" ubuntu@your-instance-public-ip
```

4. **Install the Required Libraries**

- Update the package list and install the required software:

```bash
sudo apt update
sudo apt upgrade
sudo apt install python3-pip python3-venv
# install git to clone your app
sudo apt install git
```

5. **Clone the Repository**

- Clone the repository using the following command:

```bash
git clone repository-url
```

6. **Access files using winscp and putty**
- Download and install winscp and putty to access files and terminal using GUI.
- Use putty to connect to your instance using ssh and winscp to access files.
- Use winscp to transfer files from your local machine to the instance.
- Use putty to run commands on the instance using the terminal.

7. **Install the required packages**
- Create a virtual environment and activate it:

```bash
python3 -m venv env_name
source env/bin/activate
```

## you can also use miniconda for that.

- Download mini conda.
- Install miniconda using the following command:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

`But I prefer python virtual environment for this purpose, because it is light weight and easy to use without even installing any other software.`

- Install the required packages using pip:

```bash
pip install -r requirements.txt
```

8. **Run the Webapp**

- Run the webapp using the following command by navigating to the directory where the app.py file is located:

```bash
python3 app.py
```

9. **Acces the Webapp**
- Open a web browser and go to `http://your-instance-public-ip:8501` to access the webapp.

10. **Run the Webapp in the background**

- Use the following command to run the webapp in the background:

```bash
nohup python3 app.py &
```

11. **Terminate the Webapp**

- Use the following command to terminate the webapp:

```bash
ps -aux | grep app.py
kill -9 process-id
```

12. **Access the webapp using a specific weblink other than IP**
- You can use a domain name to access the webapp by setting up a domain name and pointing it to the public IP address of your instance. You can use services like Route 53 to set up a domain name and associate it with your instance.
- You can also use a service like ngrok to create a secure tunnel to your instance and access the webapp using a specific weblink.
- Note: Make sure to secure your webapp by setting up SSL/TLS certificates and using HTTPS to encrypt the data transmitted between the client and the server.
- Note: Make sure to secure your instance by setting up a firewall, using strong passwords, and keeping the software up to date.
- Note: Make sure to monitor your instance and set up alerts to be notified of any issues or unusual activity.
- Note: Make sure to back up your data and set up automated backups to prevent data loss.
- Note: Make sure to follow best practices for security, performance, and cost optimization when deploying webapps on AWS.

13. **Terminate the instance and save money
Go to the EC2 dashboard and click on the Running Instances**
- Select the instance you want to terminate and click on the Actions button.
- Click on the Instance State option and then click on the Terminate option.
- Confirm that you want to terminate the instance.
- Note: Terminating an instance will delete all the data on the instance, so make sure to back up any data you want to keep.