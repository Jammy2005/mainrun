# since i am using runpod to access a GPU, (my laptop wasnt able to train 
# the model), runpod is already launching a docker container. So when i do 
# try to open the docker container here, i am essentially trying to open a 
# docker container within a docker container (docker in docker) which is 
# possible, however runpod blocks the operation. Hence this workaround. it 
# manually installs the environment defined in the .devcontainer/Dockerfile 
# into the existing RunPod container. The marker file is also being manually 
# created to bypass the check in utils.py

# run: chmod +x setup.sh && ./setup.sh

set -e

echo "🚀 Setting up Mainrun environment..."

# 1. Install task runner
echo "📦 Installing task runner..."
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

# 2. Install Node.js and zx
echo "📦 Installing Node.js and zx..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs
npm install -g zx

# 3. Install Python dependencies
echo "📦 Installing Python dependencies..."
# pip3 install -r ~/mainrun/.devcontainer/requirements.txt
pip3 install -r "$(pwd)/.devcontainer/requirements.txt"


# 4. Create devcontainer marker file
echo "✅ Creating devcontainer marker..."
echo "devcontainer" > /root/.mainrun

# 5. Configure git identity
echo "🔧 Configuring git identity..."
git config --global user.email "ahmad.pencil@gmail.com"
git config --global user.name "Ahmad Jamshaid"
 
echo ""
echo "✅ Setup complete! You can now run: task train"