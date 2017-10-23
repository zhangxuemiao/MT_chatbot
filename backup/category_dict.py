# coding:utf-8
def get_category_map():
    category_dict = {}
    # category_dict['alt.atheism'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # category_dict['comp.graphics'] ='00000000000000000010'
    # category_dict['comp.os.ms-windows.misc'] = '00000000000000000100'
    # category_dict['comp.sys.ibm.pc.hardware'] = '00000000000000001000'
    # category_dict['comp.sys.mac.hardware'] = '00000000000000010000'
    # category_dict['comp.windows.x'] = '00000000000000100000'
    # category_dict['misc.forsale'] = '00000000000001000000'
    # category_dict['rec.autos'] = '00000000000010000000'
    # category_dict['rec.motorcycles'] = '00000000000100000000'
    # category_dict['rec.sport.baseball'] = '00000000001000000000'
    # category_dict['rec.sport.hockey'] = '00000000010000000000'
    # category_dict['sci.crypt'] = '00000000100000000000'
    # category_dict['sci.electronics'] = '00000001000000000000'
    # category_dict['sci.med'] = '00000010000000000000'
    # category_dict['sci.space'] = '00000100000000000000'
    # category_dict['soc.religion.christian'] = '00001000000000000000'
    # category_dict['talk.politics.guns'] = '00010000000000000000'
    # category_dict['talk.politics.mideast'] = '00100000000000000000'
    # category_dict['talk.politics.misc'] = '01000000000000000000'
    # category_dict['talk.religion.misc'] = '10000000000000000000'

    category_dict['alt.atheism'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    category_dict['comp.graphics'] =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    category_dict['comp.os.ms-windows.misc'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    category_dict['comp.sys.ibm.pc.hardware'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    category_dict['comp.sys.mac.hardware'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    category_dict['comp.windows.x'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    category_dict['misc.forsale'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    category_dict['rec.autos'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    category_dict['rec.motorcycles'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['rec.sport.baseball'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['rec.sport.hockey'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['sci.crypt'] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['sci.electronics'] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['sci.med'] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['sci.space'] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['soc.religion.christian'] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['talk.politics.guns'] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['talk.politics.mideast'] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['talk.politics.misc'] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    category_dict['talk.religion.misc'] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return category_dict

if __name__=='__main__':
    category_dict = get_category_map()
    for i in category_dict:
        print(category_dict[i])