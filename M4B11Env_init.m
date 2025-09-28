function [] = M4B11Env_init(flag1,flag2,flag3)
load(fullfile(pwd,'parameterinit.mat'));
type = flag1;
amp = flag2;
comdisconnect = fix(flag3*4)+1;% 四条通讯联络线
ACgirdinit.FBUS(1) = fix(type*16) + 1; %总共16个节点
ACgirdinit.amp = (amp-0.5)*5; %±2.5的负载变化
commun = ones(1,4);
commun(comdisconnect) = 0;
save('datainit');

