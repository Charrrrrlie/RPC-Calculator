dcm_demo=dcmeci2ecef('IAU-2000/2006',[2021 5 15 17 40 53]);

ac_time=load('intermediate_res\ac_time.txt');
[L,~]=size(ac_time(:,1));
for i = 1:L
    dcm=dcmeci2ecef('IAU-2000/2006',ac_time(i,:));
    line_dcm=reshape(dcm',1,[]);  %列为主顺序展平 故先将dcm转置

    if i==1   
        fid = fopen('intermediate_res\rot.txt','w'); 
     else
         fid = fopen('intermediate_res\rot.txt','a'); 
    end
    for data = line_dcm(:)
        fprintf(fid,"%f ",data);
    end
    fprintf(fid,"\n");
    fclose(fid);
end
