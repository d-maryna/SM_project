import pandas as pd

def filter_variants(file, altfrq_threshold=5, vardiff_threshold=0.5):
    data=pd.read_excel(file)

    print('Total number of variants in file:', len(data))
    print('Allele frequency in Ukrainian population [min]:', data['Ukr_altFreq'].min())
    print('Allele frequency in Ukrainian population [median]:',data['Ukr_altFreq'].median())
    print('Allele frequency in Ukrainian population [max]:',data['Ukr_altFreq'].max())

    data['Ref_len'] = data['Ref'].apply(len)
    data['Alt_len'] = data['Alt'].apply(len)
    data['Ref_Alt_len'] = abs( data['Ref_len'] - data['Alt_len'] )
    print('Length difference btw reference and alternative allele [max]:', data['Ref_Alt_len'].max())

    print(f'Filtering variants by "Ukr_altFreq">={altfrq_threshold} and "Ref_Alt_len">={vardiff_threshold}')
    sel = data[(data['Ref_Alt_len']>=altfrq_threshold)&(data['Ukr_altFreq']>=vardiff_threshold)]

    print('Number of selected variants:', len(sel))

    return sel

def describe_vars(data):

    def get_var_type(x):
        if x['Ref_len']==x['Alt_len']:
            return 'snp'
        elif x['Ref_len']>x['Alt_len']:
            return 'del'
        elif x['Ref_len']<x['Alt_len']:
            return 'insrt'
        else:
            None

    data['var_type'] = data.apply(get_var_type, axis=1)

    print(data['var_type'].value_counts())

    print(data['Region'].value_counts())

    print(data['Associated medical condition '].value_counts())

file = './Table S5_all_differentiative_R2.xlsx'
sel = filter_variants(file)
describe_vars(sel)
sel.to_csv('selected_vars_oleksyk2021', index=False)
