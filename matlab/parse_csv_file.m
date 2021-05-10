function [params] = parse_csv_file(input_file)
%% this section reads the file into a cell array of strings
    [fileID] = fopen(input_file,'r');

    if(fileID < 0)
        disp('Error opening file.');
        return;
    end
    
    %file_line = {};
    params = [];
    idx = 1;
    while(~feof(fileID))
        temp_line = fgetl(fileID);
        if(~isempty(temp_line))
            if((temp_line(1) ~= '%') && (temp_line(1) ~= ' ') && (temp_line(1) ~= '#'))
                %file_line{idx,1} = temp_line;
                
                tmp_params = parse_csv_line(temp_line);
                for jdx=1:numel(tmp_params)
                    params(idx, jdx) = str2double(tmp_params{1, jdx});
                end
                %params(idx, :) = parse_csv_line(temp_line);
                idx = idx + 1;
            end
        end
    end
    fclose(fileID);
    
end