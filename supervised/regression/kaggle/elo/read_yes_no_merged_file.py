import pandas as panda


from dateutil import relativedelta
import datetime

def get_months_passed( given):
    
    current = datetime.datetime.now()
    given = datetime.datetime.strptime(given + '-01', '%Y-%m-%d')
    
    r = relativedelta.relativedelta(current, given)
    return r.months + r.years*12


read_yes_no_merged_data_path = 'all/yes_no_merged.csv'
yes_no_merged_data = panda.read_csv(read_yes_no_merged_data_path)
print(len(yes_no_merged_data))

##merged the total sum of purchase
total_sum = yes_no_merged_data.groupby(['card_id'])['purchase_amount'].\
                sum().reset_index().rename({'purchase_amount':'total'},axis=1)

total_sum_merged = yes_no_merged_data.merge(total_sum,how='inner',on='card_id')

# print(total_sum_merged.head(5), total_sum_merged.columns.tolist())

train_data = panda.read_csv('all/test.csv')
train_data['first_active_month'].fillna(train_data['first_active_month'].value_counts().idxmax(), inplace= True)

train_data['months_passed'] = train_data['first_active_month'].apply(lambda x: get_months_passed(x))
train_data[['card_id']] = train_data[['card_id']].apply(lambda x: x.str.upper().str.strip())

train_data = train_data[[i for i in train_data.columns.tolist() if i!='first_active_month' ]]

total_sum_merged_col = total_sum_merged[['card_id', 'total_no_of_transaction','total','total_accepted','total_rejects']]


all_merged = panda.concat([train_data, total_sum_merged_col], axis=1)
all_merged.dropna(subset=['card_id'],inplace = True)

print(len(all_merged), len(train_data))
all_merged.to_csv('all/test_all_merged.csv')
