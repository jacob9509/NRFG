function params_data = get_orl_data_2()
% this function prepares the orl dataset by removing 10% of the pixels
% uniformly randomly from each imageand returns the following:
% params_data.X: N X NxNy, where N is the number of samples, Nx and Ny is the number of pixels
% in the x and y direction of the ORL images.
% params_data.K = number of classes
% params_data.N 
% params_data.Ny 
% params_data.Nx 
% params_data.Labels: N X 1 vector of labels

K = 10;  % number of classes
Nx = 192;   % X dimension of the image
Ny = 168;  % Y dimension of th image
N = K*64;  % total number of images in the dataset used for this demo

CCC = [1:K];  % subjets from the ORL dataset used for this demo

scale = 1; % for fast computations we downsample the pixels of each image by 4

Nx = Nx*scale; % re-adjust the x-dimension of the image
Ny = Ny*scale; % re-adjust the x-dimension of the image

mask = [];   % a cell to hold the indices of the missing pixels in each image of the dataset
X = zeros(N,Nx*Ny);   % initialize the dataset with zeros.
cleandata = zeros(N,Nx*Ny);
Labels = zeros(N,1);

ll = 0; % percentage of the pixels in each image to be corrupted 0% 10% 15% 20%
in = 1;
cla = 1;
for class = CCC
    files=dir(strcat('F:\zsj\TT_PCPSF相关资料\dataset\CroppedYale\CroppedYale\yaleB',num2str(class),'\','*.pgm'));
    for n=1:N/K
        
        %read the image
        x = imread(strcat(files(n).folder,'\',files(n).name));
        x = double(x);
        %resize the image
        x = imresize(x,scale);%要归一化
        % x=(x-min(x(:)))/(max(x(:))-min(x(:)))*255;
        cleandata(in,:) = reshape(x,1,Nx*Ny);% modified
        % corrupt it with missing pixels
        rr = randi(Nx*Ny,[1,ceil((ll/100)*Nx*Ny)]);
        mask{in} = rr;
        
        x(rr) = zeros(1,length(rr));
        X(in,:) = reshape(x,1,Nx*Ny);
        Labels(in) = cla;
        in = in + 1;
        
    end
    cla = cla + 1;
end

params_data.X = X;
params_data.C = cleandata;% modified
params_data.K = K;
params_data.N = N;
params_data.Nx = Nx;
params_data.Ny = Ny;
params_data.Labels = Labels;