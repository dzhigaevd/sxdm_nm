# -*- coding: utf-8 -*-
"""
Reading, writing and modifying the information file for a typical BCDI scan and the phase retrieveal

:param:
    -pathinfor: the complete path for the aimed information filee 

:return:
    -The information file for the BCDI
"""
import os
import ast
import pandas as pd

class BCDI_Information:
    
    def __init__(self, pathinfor):
        self.pathinfor=pathinfor
        self.para_list=None
        
    def infor_writer(self):    
        """
        Saving the information file as txt file in the aimed position
        """
        list_of_lines=[]
        grouped_para=self.para_list.groupby(level='section', sort=False)
        for section, para_section in grouped_para:
            list_of_lines.append("#########%s#########\n"%section)
            for para_name, para in para_section.iterrows():
                list_of_lines.append("%s = %s\n"%(para_name[1], str(para['value'])))
            list_of_lines.append("\n")

        with open(self.pathinfor, 'w') as f:
            f.writelines(list_of_lines)
        f.close()
        return

    #reading the information file and generate the section and paralist
    def infor_reader(self):
        """
        Read the exist information file and generate the parameter list
        """
        
        assert os.path.exists(self.pathinfor), 'The information file does not exist! Please check the path again'
        index=[]
        para_value=[]
        f = open(self.pathinfor, 'r')
        for line in f:
            if line[0]=="#":
                section=line.rstrip('\n').strip('#')
            elif line!="\n":
                index.append((section, line.split(" ", 2)[0]))
                try:
                    para_value.append(ast.literal_eval(line.split(" ", 2)[-1]))    
                except:
                    para_value.append(line.split(" ", 2)[-1].rstrip())
        f.close()
        index=pd.MultiIndex.from_tuples(index, names=['section', 'paraname'])
        self.para_list=pd.DataFrame({'value':para_value}, index=index)
        return
    
    def get_para_value(self, para_name, section=''):
        """
        Get the parameter value from the parameter list, the section name can be provided if the parameter is not unique
        
        :Input:
            -para_name: the name of the parameter
            -section (optional): the section name to specify the non-unique parameter
        :return:
            -the value of aimed parameter if it exists
        """

        if para_name not in self.para_list.index.get_level_values('paraname'):
            print('Could not find the desired parameter in the information file! Please check the parameter name again!')
            return
        elif list(self.para_list.index.get_level_values('paraname')).count(para_name)>1 and section=='':
            print('More than one paramter with the same name! Please specify the sections of the desired parameter!')
            return
        elif section=='':
            return self.para_list.xs(para_name, level='paraname').iloc[0,0]
        else:
            return self.para_list.loc[(section, para_name), 'value']
    
    def add_para(self, para_name, section, para_value):
        """
        Add the parameter value to the parameter list
        
        :Input:
            -para_name: the name of the parameter
            -section: the section name to specify the non-unique parameter
            -para: the parameter value
        """
        if self.para_list is None:
            index=pd.MultiIndex.from_tuples([(section, para_name)], names=['section', 'paraname'])
            self.para_list= pd.DataFrame({'value':str(para_value)}, index=index)
        elif (section, para_name) not in self.para_list.index:
            index=pd.MultiIndex.from_tuples([(section, para_name)], names=['section', 'paraname'])
            self.para_list=pd.concat([self.para_list, pd.DataFrame({'value':str(para_value)}, index=index)])
        else:
            self.para_list.at[(section, para_name), 'value']=para_value
        return
    

def main():
    path=r'E:\Work place 2\sample\XRD\20190530 Desy P10\scan file 2\Area D2\scan00028\cutqx\2D_retrieval_information.txt'
    infor=BCDI_Information(path)
    infor.infor_reader()
    infor.add_para('size_lim', "Trial 04", [28, 29])
    print(infor.para_list.index)
    return


if __name__=='__main__':
    main()