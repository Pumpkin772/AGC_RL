function [] = M10B39Env_init(flag1,flag2,flag3)
load(fullfile(pwd,'parameterinit.mat'));
type = flag1;
amp = flag2;
comdisconnect = fix(flag3*17);% 十六条通讯联络线，一共17种工况
loadnode = fix(type*39) + 1; %总共16个节点
loadsize = (amp-0.5)*40; %±20的负载变化
commun = ones(8,2);
if comdisconnect~=16
    comdisconnectT = mod(comdisconnect,8)+1;
    comdisconnectR = fix(comdisconnect/8)+1;
    commun(comdisconnectT,comdisconnectR) = 0;
end
save('datainit');

