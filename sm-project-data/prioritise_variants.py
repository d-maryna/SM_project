import sys
import pandas as pd

def filter_variants(file, min_len_diff=5, min_ukr_altfreq=0.5):
    data = pd.read_excel(file)
    data.columns = data.columns.str.strip()  # in the xlsx some names end with a space
    print('Total number of variants in file:', len(data))
    print('Allele frequency in Ukrainian population [min]:', data['Ukr_altFreq'].min())
    print('Allele frequency in Ukrainian population [median]:', data['Ukr_altFreq'].median())
    print('Allele frequency in Ukrainian population [max]:', data['Ukr_altFreq'].max())
    data['Ref_len'] = data['Ref'].apply(len)
    data['Alt_len'] = data['Alt'].apply(len)
    data['Ref_Alt_len'] = abs(data['Ref_len'] - data['Alt_len'])
    print('Length difference btw reference and alternative allele [max]:', data['Ref_Alt_len'].max())
    print(f'Filtering variants by "Ref_Alt_len">={min_len_diff} and "Ukr_altFreq">={min_ukr_altfreq}')
    sel = data[(data['Ref_Alt_len'] >= min_len_diff) & (data['Ukr_altFreq'] >= min_ukr_altfreq)]
    print('Number of selected variants:', len(sel))
    return sel.copy()

def describe_vars(data):
    def get_var_type(x):
        if x['Ref_len'] == x['Alt_len']:
            return 'snp'
        elif x['Ref_len'] > x['Alt_len']:
            return 'del'
        else:
            return 'insrt'
    data['var_type'] = data.apply(get_var_type, axis=1)
    print(data['var_type'].value_counts())
    print(data['Region'].value_counts())
    print(data['Associated medical condition'].value_counts())

def orca_subset(sel):
    # intergenic+ncRNA (all subtypes); '.' (no annotation) is counted as intergenic for now
    region = sel['Region'].astype(str)
    mask = region.isin(['intergenic', '.']) | region.str.startswith('ncRNA')
    sub = sel[mask].copy()
    print('Number of variants in Orca subset (intergenic, ".", ncRNA*):', len(sub))
    return sub

if __name__ == '__main__':
    file = sys.argv[1] if len(sys.argv) > 1 else './Table S5_all_differentiative_R2.xlsx'
    sel = filter_variants(file)
    describe_vars(sel)
    sel.to_csv('selected_vars_oleksyk2021.csv', index=False)
    orca_subset(sel).to_csv('selected_vars_oleksyk2021_orca_subset.csv', index=False)