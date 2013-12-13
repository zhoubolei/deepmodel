DECONV ='/home/Stephen/Desktop/VisionLib/Donglai/DeepL/DeconvNetToolbox/';
addpath(DECONV)
setupDeconvNetToolbox
DD=pwd;
eval(['cd ' DECONV 'PoolingToolbox/']) 
compilemex

%{
% ipp
eval(['cd ' DECONV 'IPPConvsToolbox/MEX']) 
compilemex
% gpu
%}