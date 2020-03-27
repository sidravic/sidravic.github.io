# Predicting the outcome of EPL fixtures
> Tabular data predictions


```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
%reload_ext autoreload
```

```python
from fastai import *
from fastai.tabular import *
```

## Goal

The goal of this exercise was to be able to predict premier league match results based on historical performances of teams with reasonable accuracy using [the paper by Geetanjali Tewari and Krishna Kartik Darsipudi](https://github.com/krishnakartik1/LSTM-footballMatchWinner/blob/master/Set_Paper.pdf) as a reference using Fastai.

I pulled the data for this exercise from [footystats.org](https://footystats.org/download-stats-csv#) and [football-data](https://www.football-data.co.uk/)



### Datasets


The dataset is from the English premier league for the last 10 seasons. However, well use just the last 3 seasons to predict the outcome of 40 matches from season 2018-2019. 

More recent datasets are available on footystats.org with subscriptions. 

```python
DATA_PATH = Path('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data')
```

```python
DATA_PATH.ls()
```




    [PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1213_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1415_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-0910_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1617_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-0910_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1516_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1617_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1314_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1718_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1516_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1011_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1415_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1718_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1011_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/validation_report.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1112_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1213_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1819_json.json'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1819_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1112_csv.csv'),
     PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/footystats/epl/data/season-1314_json.json')]



### Summary of the dataset


A quick peak at the dataset shows 380 rows for a season with 62 columns. I've got a lot of features and it would make sense to use key map to quickly get an idea of what the fields indicate.

A sample of the season 2018-19 looks like this.



```python
s1819_df = pd.read_csv(DATA_PATH/'season-1819_csv.csv')
s1819_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Div</th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HTR</th>
      <th>...</th>
      <th>BbAv&lt;2.5</th>
      <th>BbAH</th>
      <th>BbAHh</th>
      <th>BbMxAHH</th>
      <th>BbAvAHH</th>
      <th>BbMxAHA</th>
      <th>BbAvAHA</th>
      <th>PSCH</th>
      <th>PSCD</th>
      <th>PSCA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E0</td>
      <td>10/08/2018</td>
      <td>Man United</td>
      <td>Leicester</td>
      <td>2</td>
      <td>1</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>1.79</td>
      <td>17</td>
      <td>-0.75</td>
      <td>1.75</td>
      <td>1.70</td>
      <td>2.29</td>
      <td>2.21</td>
      <td>1.55</td>
      <td>4.07</td>
      <td>7.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Bournemouth</td>
      <td>Cardiff</td>
      <td>2</td>
      <td>0</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>1.83</td>
      <td>20</td>
      <td>-0.75</td>
      <td>2.20</td>
      <td>2.13</td>
      <td>1.80</td>
      <td>1.75</td>
      <td>1.88</td>
      <td>3.61</td>
      <td>4.70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Fulham</td>
      <td>Crystal Palace</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>...</td>
      <td>1.87</td>
      <td>22</td>
      <td>-0.25</td>
      <td>2.18</td>
      <td>2.11</td>
      <td>1.81</td>
      <td>1.77</td>
      <td>2.62</td>
      <td>3.38</td>
      <td>2.90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Huddersfield</td>
      <td>Chelsea</td>
      <td>0</td>
      <td>3</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>...</td>
      <td>1.84</td>
      <td>23</td>
      <td>1.00</td>
      <td>1.84</td>
      <td>1.80</td>
      <td>2.13</td>
      <td>2.06</td>
      <td>7.24</td>
      <td>3.95</td>
      <td>1.58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Newcastle</td>
      <td>Tottenham</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>...</td>
      <td>1.81</td>
      <td>20</td>
      <td>0.25</td>
      <td>2.20</td>
      <td>2.12</td>
      <td>1.80</td>
      <td>1.76</td>
      <td>4.74</td>
      <td>3.53</td>
      <td>1.89</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>375</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Liverpool</td>
      <td>Wolves</td>
      <td>2</td>
      <td>0</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>2.31</td>
      <td>22</td>
      <td>-1.50</td>
      <td>1.98</td>
      <td>1.91</td>
      <td>2.01</td>
      <td>1.95</td>
      <td>1.32</td>
      <td>5.89</td>
      <td>9.48</td>
    </tr>
    <tr>
      <th>376</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Man United</td>
      <td>Cardiff</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>...</td>
      <td>2.95</td>
      <td>21</td>
      <td>-2.00</td>
      <td>2.52</td>
      <td>2.32</td>
      <td>1.72</td>
      <td>1.64</td>
      <td>1.30</td>
      <td>6.06</td>
      <td>9.71</td>
    </tr>
    <tr>
      <th>377</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Southampton</td>
      <td>Huddersfield</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>2.29</td>
      <td>22</td>
      <td>-1.50</td>
      <td>2.27</td>
      <td>2.16</td>
      <td>1.80</td>
      <td>1.73</td>
      <td>1.37</td>
      <td>5.36</td>
      <td>8.49</td>
    </tr>
    <tr>
      <th>378</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Tottenham</td>
      <td>Everton</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>2.07</td>
      <td>19</td>
      <td>-0.50</td>
      <td>2.13</td>
      <td>2.08</td>
      <td>1.85</td>
      <td>1.80</td>
      <td>1.91</td>
      <td>3.81</td>
      <td>4.15</td>
    </tr>
    <tr>
      <th>379</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Watford</td>
      <td>West Ham</td>
      <td>1</td>
      <td>4</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>...</td>
      <td>2.44</td>
      <td>19</td>
      <td>-0.50</td>
      <td>2.25</td>
      <td>2.19</td>
      <td>1.78</td>
      <td>1.72</td>
      <td>2.11</td>
      <td>3.86</td>
      <td>3.41</td>
    </tr>
  </tbody>
</table>
<p>380 rows × 62 columns</p>
</div>



### Getting rid of bookie data attributes

Each season contains a lot of bookie related information which is not directly relevant to the game so I'll ignore those. However, it's an interesting exercise to determine which bets would be the most rewarding but I'm going to skip that for now.

I'm going to use the key file available [here](https://www.football-data.co.uk/notes.txt) to filter out columns that are relevant to the game. 

I've also create a few helper methods to parse out text and fix the date formats in the different CSV files. The date format in the 2018-2019 season is the `DD-MM-YYYY` format while the others use `DD-MM-YY` format. So fixing this upfront allows us to use the `Date` attribute to build a sorted by time dataframe. 

This is relevant as the form of a team is calculated based on their last few performances.


```python
cols = """
Div = League Division
Date = Match Date (dd/mm/yy)
Time = Time of match kick off
HomeTeam = Home Team
AwayTeam = Away Team
FTHG and HG = Full Time Home Team Goals
FTAG and AG = Full Time Away Team Goals
FTR and Res = Full Time Result (H=Home Win, D=Draw, A=Away Win)
HTHG = Half Time Home Team Goals
HTAG = Half Time Away Team Goals
HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)
Attendance = Crowd Attendance
Referee = Match Referee
HS = Home Team Shots
AS = Away Team Shots
HST = Home Team Shots on Target
AST = Away Team Shots on Target
HHW = Home Team Hit Woodwork
AHW = Away Team Hit Woodwork
HC = Home Team Corners
AC = Away Team Corners
HF = Home Team Fouls Committed
AF = Away Team Fouls Committed
HFKC = Home Team Free Kicks Conceded
AFKC = Away Team Free Kicks Conceded
HO = Home Team Offsides
AO = Away Team Offsides
HY = Home Team Yellow Cards
AY = Away Team Yellow Cards
HR = Home Team Red Cards
AR = Away Team Red Cards
HBP = Home Team Bookings Points (10 = yellow, 25 = red)
ABP = Away Team Bookings Points (10 = yellow, 25 = red)
"""

def split_lines(cols_str): return cols_str.split("\n")
def filter_blanks(cols:list): return list(filter(None, cols))      
def split_item(item): return item.split(" = ", maxsplit=1)
def trim_item(val): return (val[0].strip(), val[1].strip())


def get_valid_cols(cols):
    valid_cols = split_lines(cols)
    valid_cols = filter_blanks(valid_cols)
    valid_cols = [tuple(split_item(item))  for item in valid_cols ]    
    valid_cols = dict(list(map(trim_item, valid_cols)))
    return valid_cols
    
all_cols = get_valid_cols(cols)
all_cols
```




    {'Div': 'League Division',
     'Date': 'Match Date (dd/mm/yy)',
     'Time': 'Time of match kick off',
     'HomeTeam': 'Home Team',
     'AwayTeam': 'Away Team',
     'FTHG and HG': 'Full Time Home Team Goals',
     'FTAG and AG': 'Full Time Away Team Goals',
     'FTR and Res': 'Full Time Result (H=Home Win, D=Draw, A=Away Win)',
     'HTHG': 'Half Time Home Team Goals',
     'HTAG': 'Half Time Away Team Goals',
     'HTR': 'Half Time Result (H=Home Win, D=Draw, A=Away Win)',
     'Attendance': 'Crowd Attendance',
     'Referee': 'Match Referee',
     'HS': 'Home Team Shots',
     'AS': 'Away Team Shots',
     'HST': 'Home Team Shots on Target',
     'AST': 'Away Team Shots on Target',
     'HHW': 'Home Team Hit Woodwork',
     'AHW': 'Away Team Hit Woodwork',
     'HC': 'Home Team Corners',
     'AC': 'Away Team Corners',
     'HF': 'Home Team Fouls Committed',
     'AF': 'Away Team Fouls Committed',
     'HFKC': 'Home Team Free Kicks Conceded',
     'AFKC': 'Away Team Free Kicks Conceded',
     'HO': 'Home Team Offsides',
     'AO': 'Away Team Offsides',
     'HY': 'Home Team Yellow Cards',
     'AY': 'Away Team Yellow Cards',
     'HR': 'Home Team Red Cards',
     'AR': 'Away Team Red Cards',
     'HBP': 'Home Team Bookings Points (10 = yellow, 25 = red)',
     'ABP': 'Away Team Bookings Points (10 = yellow, 25 = red)'}



```python
betting_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD',
 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD',
 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH',
 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'PSCH', 'PSCD', 'PSCA']
```

### Creating a list of bookie attributes columns

So I've created a list of columns that are bookie attributes and we simply filter those out from the dataframe. Nothing fancy here.

The `get_col_name` is a simple helper function to quickly lookup the column's name from the abbreviation. This comes in handy when defining continuous and categorical variables.

```python
def get_col_name(col_abbr): 
    name = []
    key_vals = []
    col_name = all_cols.get(col_abbr, None)   
    
    if col_name:
        key_vals.append(col_abbr)
        name.append(col_name)
    else:
        possible_cols = list(filter(lambda s:s.startswith(col_abbr), all_cols.keys()))        
        [name.append(all_cols[key]) for key in possible_cols]
        key_vals = key_vals + possible_cols    
    return name, key_vals

def remove_betting_cols(df):
    return df.loc[:, ~df.columns.isin(betting_cols)]

```

```python
get_col_name('HST')
```




    (['Home Team Shots on Target'], ['HST'])



### Build Data set of 3 seasons

1. Uniform date format for all seasons
2. Append all seasons info into a single dataframe
3. Compute winning streaks

This is actualy code for loading the dataset and fixing the date format. I'll improve this further but for now this creates a single dataframe with information loaded from 3 seasons. 

```python
def fix_date(d, col='Date'): 
    def _fix_date(d):
        day, m, y = d.split('/')
        if len(y) == 2: return f'{day}/{m}/20{y}' 
        else: return d
    
    d[col] = d[col].apply(_fix_date)
    return d
    

def build_data_set(seasons=['1617', '1718', '1819']):
    df = pd.DataFrame()
    for season in seasons:
        df_ = pd.read_csv(DATA_PATH/f'season-{season}_csv.csv')
        df_ = remove_betting_cols(df_) 
        df_ = fix_date(df_, 'Date')  
        df_['Date'] = df_['Date'].astype('datetime64[ns]')
        df = df.append(df_)
        
    df.sort_values(by='Date',ascending=True, inplace=True)
    return df

    

data_df = build_data_set()
data_df
    
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Div</th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HTR</th>
      <th>...</th>
      <th>AF</th>
      <th>HC</th>
      <th>AC</th>
      <th>HY</th>
      <th>AY</th>
      <th>HR</th>
      <th>AR</th>
      <th>LBH</th>
      <th>LBD</th>
      <th>LBA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>Hull</td>
      <td>Chelsea</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>...</td>
      <td>15</td>
      <td>5</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>7.00</td>
      <td>4.40</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>63</th>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>Swansea</td>
      <td>Liverpool</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>9</td>
      <td>3</td>
      <td>10</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>7.50</td>
      <td>4.75</td>
      <td>1.44</td>
    </tr>
    <tr>
      <th>64</th>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>Watford</td>
      <td>Bournemouth</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>...</td>
      <td>12</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>2.38</td>
      <td>3.25</td>
      <td>3.25</td>
    </tr>
    <tr>
      <th>65</th>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>West Ham</td>
      <td>Middlesbrough</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>...</td>
      <td>12</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2.20</td>
      <td>3.40</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>62</th>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>Sunderland</td>
      <td>West Brom</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>...</td>
      <td>13</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2.40</td>
      <td>3.20</td>
      <td>3.25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>375</th>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Liverpool</td>
      <td>Wolves</td>
      <td>2</td>
      <td>0</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>376</th>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Man United</td>
      <td>Cardiff</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>...</td>
      <td>6</td>
      <td>11</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>377</th>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Southampton</td>
      <td>Huddersfield</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>378</th>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Tottenham</td>
      <td>Everton</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>13</td>
      <td>7</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>379</th>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Watford</td>
      <td>West Ham</td>
      <td>1</td>
      <td>4</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>...</td>
      <td>10</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1140 rows × 26 columns</p>
</div>



### Inferred attributes 

Now based on the paper there are certain attributes that represent the teams current form and the handicaps with which the team is playing. 

The document highlights attributes for a team such as 
1. 3 game winning and losing streaks 
2. 5 game winning and losing streaks 
3. 3 game home winning streaks
4. 5 game home winning streaks
5. points accummalated in the last 4 games
6. Number of yellow cards and reds cards
7. Number of corners secured
8. Half time goals scored among many others.

So we loop through each seasons document and aggregate this information and write the teams data onto a file. 

The long and unwieldy `compute_season` method contains the meat of the computation but to follow the code I'd recommend beginning at the bottom at the `prepare_model_data` function.


```python
def fix_date(d, col='Date'): 
    def _fix_date(d):
        day, m, y = d.split('/')
        if len(y) == 2: return f'{day}/{m}/20{y}' 
        else: return d
    
    d[col] = d[col].apply(_fix_date)
    return d
    
def build_season(season_df, season_year, debug=False):    
    for team in get_teams(season_df):
        if debug: print(f'Preparing {team} for {season_year}')
        compute_season_for(season_df, team, season_year,storage_path=TEAM_SEASON_PATH,debug=debug)    
    return

def collate(season_year, storage_path=TEAM_SEASON_PATH, debug=False):
    df = pd.DataFrame()
    if debug: print(f'Collating season for {season_year}')        
    for file in glob.glob(f'{storage_path}/*_{season_year}.csv'):
        df_ = pd.read_csv(file)
        df = df.append(df_)    
    df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)
    df.sort_values(by='Date', inplace=True, ascending=True)
    return df

def load_season(season, path=DATA_PATH, debug=False):
    if debug: print(f'Loading files for season {season}')
    df_ = pd.read_csv(path/f'season-{season}_csv.csv')
    df_ = remove_betting_cols(df_) 
    df_ = fix_date(df_, 'Date')  
    df_['Date'] = df_['Date'].astype('datetime64[ns]')  
    df_.sort_values(by='Date', ascending=True, inplace=True)
    return df_
    
def build_data_set(seasons=['1617'], storage_path=TEAM_SEASON_PATH, debug=False):
    for season in seasons:        
        df_ = load_season(season, path=DATA_PATH, debug=debug)        
        build_season(season_df=df_, season_year=season, debug=debug)           
        
    df = pd.DataFrame()
    for season in seasons: df = df.append(collate(season))      
    return df

def prepare_model_data(path, debug=True):
    df = build_data_set(seasons=['1617', '1718', '1819'], 
                     storage_path=TEAM_SEASON_PATH, debug=debug)
    
    df.to_csv(path/'model_data.csv')
    if debug: print(f'Model data saved at {path.__str__()}/model_data.csv')
    return df
```

1. This starts the process of creating the final dataset we'll use to train our model. It calls the `build_data_set` method passing the seasons and a location to store the files it generates for each team per season. 

2. `build_data_set` loads the data from csvs, builds the season's inferred attributes for each team (which is written to a file) and then once all the csvs for every team is generated it starts the collating (`collate`) the data and building a single dataframe which is returned to the `prepare_model_data` and persisted to another folder.

3. the `load_season` function does the date fixing for us but now for all season csv we load and calls the `build_season` method. `build_season` is just a iteration manager method which calls the `compute_season` where we generate all the inferred attributes.



```python
sdf = build_data_set(seasons=['1819'])
```


### Computing inferred attributes

The methods below compute the inferred attributes and the names are fairly indicative of the function they perform. 

I've created a few helper methods to work with individual teams and seasons such as 

1. `get_teams` - returns all the teams playing in league for that season
2. `get_team_seasons` - returns the matches for a specific team for a specific season. 
3. `match_for` - is just a generator function which makes it easy to loop through the games in each season.
4. `game_location` - tells us if the team (in consideration) played the match as an "Away team" or the "Home team".
5. `get_result` is a helper that determines if a game was won or lost (for the team under consideration)
6. `goal_difference` - calculates goal difference
7. `get_streak` - is a closure which is used to compute different kinds of streaks (3,5 win/loss, home-win/home-loss, away-win/away-loss) streaks. We use this in the compute method.
8. `matches_before` - returns the matches for a team before a specific date. This is needed to compute the points earned in the last 4 matches and is useful in determining the form in which the team is.

```python
def get_teams(df): return sorted(set(df['HomeTeam'].values))
def get_team_season(team, df):
    return df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team))].copy()

def matches_for(team, df):
    team_season_df = get_team_season(team, df)
    for idx, row in team_season_df.iterrows():
        yield team, row

def game_location(team, match): 
    return 'A' if match.AwayTeam == team else 'H'
    
def get_result(team, match):
    if match.FTR == "D": return match.FTR
    elif ((match.FTR == "A" and match.AwayTeam == team) or 
         (match.FTR == "H" and match.HomeTeam == team)): return "W"
    else: return "L"
        
def goal_difference(game_loc, match):
    if game_loc == 'A':
        return ((match.FTAG - match.FTHG), (match.HTAG - match.HTHG))        
    else: 
        return ((match.FTHG - match.FTAG), (match.HTHG - match.HTAG))         

def get_streak(streak, streak_size, win_loss):
    def win_streak(streak): 
        return ((len(set(streak))==1) and streak[-1] == 'W' and len(streak) == streak_size)
    def loss_streak(streak): 
        return ((len(set(streak))==1) and streak[-1] == 'L' and len(streak) == streak_size)
        
    if len(streak) == 5: streak.pop(0)
        
    streak.append(win_loss)
    w = win_streak(streak[-1 * streak_size:])       
    l = loss_streak(streak[-1 * streak_size:])
        
    return (streak, w, l)
    
def matches_before_df(date, df):
    return df[df['Date'] < date]

def last_4_games_points(team, date, df):
    team_df = get_team_season(team, df)
    matches_df = matches_before_df(date, team_df).iloc[-4:]
    pts = 0
    for match in matches_df.iloc:    
        result = match.FTR        
        if result == 'D': pts += 1
        if result == 'A' and match.AwayTeam == 'Arsenal': pts += 3
        if result == 'H' and match.HomeTeam == 'Arsenal': pts += 3
    return pts
    

```

```python
TEAM_SEASON_PATH = Path('/home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/team_season_data')

def compute_season_for(season_df, tm, season_year, storage_path=TEAM_SEASON_PATH, debug=False):    
    results = []
    game_loc = []    
    ft_goal_difference = []
    ht_goal_difference = []  
    
    if debug: print(f'Computing season for {tm}')
        
    win3_counter, loss3_counter = 0, 0
    win5_counter, loss5_counter = 0, 0
    home_win3_counter, home_loss3_counter = 0,0
    home_win5_counter, home_loss5_counter = 0,0
    away_win3_counter, away_loss3_counter = 0,0
    away_win5_counter, away_loss5_counter = 0,0

    win_3_streak, loss_3_streak = [], []
    win_5_streak, loss_5_streak = [], []
    home_win_3_streak, home_loss_3_streak = [], []
    home_win_5_streak, home_loss_5_streak = [], []
    away_win_5_streak, away_loss_5_streak = [], []
    away_win_3_streak, away_loss_3_streak = [], []

    win3 = partial(get_streak, [], 3)
    win5 = partial(get_streak, [], 5)

    home_win3 = partial(get_streak, [], 3)
    away_win3 = partial(get_streak, [], 3)
    home_win5 = partial(get_streak, [], 5)
    away_win5 = partial(get_streak, [], 5)

    home_last4pts = []
    away_last4pts = []    
    
    for team, match in matches_for(tm, season_df):
        r = get_result(team, match)
        
        results.append(r)        
        game_loc.append(game_location(team, match))
        ft_goal_diff, ht_goal_diff = goal_difference(game_loc[-1], match)
        ft_goal_difference.append(ft_goal_diff)           
        ht_goal_difference.append(ht_goal_diff)                   

        _, win3_streak, loss3_streak = win3(r)
        if win3_streak: win3_counter += 1 
        if loss3_streak: loss3_counter += 1      
        win_3_streak.append(win3_counter)
        loss_3_streak.append(loss3_counter)

        _, win5_streak, loss5_streak = win5(r)
        if win5_streak: win5_counter += 1
        if loss5_streak: loss5_counter += 1
        win_5_streak.append(win5_counter)
        loss_5_streak.append(loss5_counter)            

        # away win streak
        if game_loc == 'A':
            _, away_win3_streak, away_loss3_streak = away_win3(r)
            if away_win3_streak: away_win3_counter += 1
            if away_loss3_streak: away_loss3_counter +=1

            _, away_win5_streak, away_loss5_streak = away_win5(r)
            if away_win5_streak: away_win5_counter += 1
            if away_loss5_streak: away_loss5_counter +=1       
        else:
            _, home_win3_streak, home_loss3_streak = home_win3(r)
            if home_win3_streak: home_win3_counter += 1
            if home_loss3_streak: home_loss3_counter +=1

            _, home_win5_streak, home_loss5_streak = home_win5(r)
            if home_win5_streak: home_win5_counter += 1
            if home_loss5_streak: home_loss5_counter +=1

        away_win_3_streak.append(away_win3_counter)            
        away_loss_3_streak.append(away_loss3_counter)               
        away_win_5_streak.append(away_win5_counter)            
        away_loss_5_streak.append(away_loss5_counter)
        home_win_3_streak.append(home_win3_counter)            
        home_loss_3_streak.append(home_loss3_counter)
        home_win_5_streak.append(home_win5_counter)            
        home_loss_5_streak.append(home_loss5_counter)

        # points in the last 4 games
        home_last4pts.append(last_4_games_points(match.HomeTeam, match.Date, season_df))
        away_last4pts.append(last_4_games_points(match.AwayTeam, match.Date, season_df))

    team_season_df = get_team_season(tm, season_df)    
    team_season_df['game_loc'] = game_loc
    team_season_df['ht_goal_difference'] = ht_goal_difference
    team_season_df['ft_goal_difference'] = ft_goal_difference
    team_season_df['win3_streak'] = win_3_streak
    team_season_df['loss3_streak'] = loss_3_streak
    team_season_df['win5_streak'] = win_5_streak
    team_season_df['loss5_streak'] = loss_5_streak
    team_season_df['away_win3_streak'] = away_win_3_streak
    team_season_df['away_win5_streak'] = away_win_5_streak
    team_season_df['away_loss3_streak'] = away_loss_3_streak
    team_season_df['away_loss5_streak'] = away_loss_5_streak
    team_season_df['home_win3_streak'] = home_win_3_streak
    team_season_df['home_win5_streak'] = home_win_5_streak
    team_season_df['home_loss3_streak'] = home_loss_3_streak
    team_season_df['home_loss5_streak'] = home_loss_5_streak
    team_season_df['home_last4pts'] = home_last4pts
    team_season_df['away_last4pts'] = away_last4pts
    
    if debug: print(f'Saved CSV for {tm}_{season_year}.csv')
    team_season_df.to_csv(f'{TEAM_SEASON_PATH/tm}_{season_year}.csv')
    return


```

```python
def fix_date(d, col='Date'): 
    def _fix_date(d):
        day, m, y = d.split('/')
        if len(y) == 2: return f'{day}/{m}/20{y}' 
        else: return d
    
    d[col] = d[col].apply(_fix_date)
    return d
    
def build_season(season_df, season_year, debug=False):    
    for team in get_teams(season_df):
        if debug: print(f'Preparing {team} for {season_year}')
        compute_season_for(season_df, team, season_year,storage_path=TEAM_SEASON_PATH,debug=debug)    
    return

def collate(season_year, storage_path=TEAM_SEASON_PATH, debug=False):
    df = pd.DataFrame()
    if debug: print(f'Collating season for {season_year}')        
    for file in glob.glob(f'{storage_path}/*_{season_year}.csv'):
        df_ = pd.read_csv(file)
        df = df.append(df_)    
    df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)
    df.sort_values(by='Date', inplace=True, ascending=True)
    return df

def load_season(season, path=DATA_PATH, debug=False):
    if debug: print(f'Loading files for season {season}')
    df_ = pd.read_csv(path/f'season-{season}_csv.csv')
    df_ = remove_betting_cols(df_) 
    df_ = fix_date(df_, 'Date')  
    df_['Date'] = df_['Date'].astype('datetime64[ns]')  
    df_.sort_values(by='Date', ascending=True, inplace=True)
    return df_
    
def build_data_set(seasons=['1617'], storage_path=TEAM_SEASON_PATH, debug=False):
    for season in seasons:        
        df_ = load_season(season, path=DATA_PATH, debug=debug)        
        build_season(season_df=df_, season_year=season, debug=debug)           
        
    df = pd.DataFrame()
    for season in seasons: df = df.append(collate(season))      
    return df



def prepare_model_data(path, debug=True):
    df = build_data_set(seasons=['1617', '1718', '1819'], 
                     storage_path=TEAM_SEASON_PATH, debug=debug)
    
    df.to_csv(path/'model_data.csv')
    if debug: print(f'Model data saved at {path.__str__()}/model_data.csv')
    return df
    


```

```python
MODEL_DATA_PATH = Path('/home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data')
xdf = prepare_model_data(MODEL_DATA_PATH, debug=True)
xdf
```

    Loading files for season 1617
    Preparing Arsenal for 1617
    Computing season for Arsenal
    Saved CSV for Arsenal_1617.csv
    ...
    Preparing Wolves for 1819
    Computing season for Wolves
    Saved CSV for Wolves_1819.csv
    Model data saved at /home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data/model_data.csv


Time to train.

```python
from fastai import *
```

```python
from fastai.tabular import *
```

## Train

I then load the file generated as part of our preparation process. Since I generated our csv for training by generating csvs for each team for each season it was necessary to ensure I didn't have duplicates. 




```python
MODEL_DATA_PATH = Path('/home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data')
```

```python
df = pd.read_csv(MODEL_DATA_PATH/'model_data.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>Div</th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>...</th>
      <th>away_win3_streak</th>
      <th>away_win5_streak</th>
      <th>away_loss3_streak</th>
      <th>away_loss5_streak</th>
      <th>home_win3_streak</th>
      <th>home_win5_streak</th>
      <th>home_loss3_streak</th>
      <th>home_loss5_streak</th>
      <th>home_last4pts</th>
      <th>away_last4pts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>62</td>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>Sunderland</td>
      <td>West Brom</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>65</td>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>West Ham</td>
      <td>Middlesbrough</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>63</td>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>Swansea</td>
      <td>Liverpool</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>64</td>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>Watford</td>
      <td>Bournemouth</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>61</td>
      <td>E0</td>
      <td>2016-01-10</td>
      <td>Hull</td>
      <td>Chelsea</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>37</td>
      <td>373</td>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Fulham</td>
      <td>Newcastle</td>
      <td>0</td>
      <td>4</td>
      <td>A</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>37</td>
      <td>376</td>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Man United</td>
      <td>Cardiff</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>37</td>
      <td>377</td>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Southampton</td>
      <td>Huddersfield</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>37</td>
      <td>379</td>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Watford</td>
      <td>West Ham</td>
      <td>1</td>
      <td>4</td>
      <td>A</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>37</td>
      <td>378</td>
      <td>E0</td>
      <td>2019-12-05</td>
      <td>Tottenham</td>
      <td>Everton</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1140 rows × 45 columns</p>
</div>



```python
df.columns
```




    Index(['Unnamed: 0', 'Unnamed: 0.1', 'Div', 'Date', 'HomeTeam', 'AwayTeam',
           'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS',
           'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'LBH',
           'LBD', 'LBA', 'game_loc', 'ht_goal_difference', 'ft_goal_difference',
           'win3_streak', 'loss3_streak', 'win5_streak', 'loss5_streak',
           'away_win3_streak', 'away_win5_streak', 'away_loss3_streak',
           'away_loss5_streak', 'home_win3_streak', 'home_win5_streak',
           'home_loss3_streak', 'home_loss5_streak', 'home_last4pts',
           'away_last4pts'],
          dtype='object')



Some basic cleanup in getting rid of default columns generated by pandas.

```python
df.drop(labels=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True, axis=1)
df.drop(labels=['Div'], axis=1, inplace=True)
df
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HTR</th>
      <th>Referee</th>
      <th>...</th>
      <th>away_win3_streak</th>
      <th>away_win5_streak</th>
      <th>away_loss3_streak</th>
      <th>away_loss5_streak</th>
      <th>home_win3_streak</th>
      <th>home_win5_streak</th>
      <th>home_loss3_streak</th>
      <th>home_loss5_streak</th>
      <th>home_last4pts</th>
      <th>away_last4pts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-10</td>
      <td>Sunderland</td>
      <td>West Brom</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>S Attwell</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-10</td>
      <td>West Ham</td>
      <td>Middlesbrough</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>N Swarbrick</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-10</td>
      <td>Swansea</td>
      <td>Liverpool</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>M Oliver</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-10</td>
      <td>Watford</td>
      <td>Bournemouth</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>M Dean</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-10</td>
      <td>Hull</td>
      <td>Chelsea</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>A Taylor</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>2019-12-05</td>
      <td>Fulham</td>
      <td>Newcastle</td>
      <td>0</td>
      <td>4</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>K Friend</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>2019-12-05</td>
      <td>Man United</td>
      <td>Cardiff</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>J Moss</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>2019-12-05</td>
      <td>Southampton</td>
      <td>Huddersfield</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>L Probert</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>2019-12-05</td>
      <td>Watford</td>
      <td>West Ham</td>
      <td>1</td>
      <td>4</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>C Kavanagh</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>2019-12-05</td>
      <td>Tottenham</td>
      <td>Everton</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>A Marriner</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1140 rows × 42 columns</p>
</div>



```python
df.columns
```




    Index(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
           'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
           'HY', 'AY', 'HR', 'AR', 'LBH', 'LBD', 'LBA', 'game_loc',
           'ht_goal_difference', 'ft_goal_difference', 'win3_streak',
           'loss3_streak', 'win5_streak', 'loss5_streak', 'away_win3_streak',
           'away_win5_streak', 'away_loss3_streak', 'away_loss5_streak',
           'home_win3_streak', 'home_win5_streak', 'home_loss3_streak',
           'home_loss5_streak', 'home_last4pts', 'away_last4pts'],
          dtype='object')



### Splitting date attributes into continuous and categorical attributes. 


This adds a lot of meaningful information on things like was the team playing well when they had to play a midweek game among other things. 

This would be even more valuable if we added the Champions leagues and Europa league games the top clubs are involved in but I'm going to stick with this for now.

Notice the column count went up.

```python
add_datepart(df, 'Date', drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HTR</th>
      <th>Referee</th>
      <th>HS</th>
      <th>...</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sunderland</td>
      <td>West Brom</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>S Attwell</td>
      <td>7</td>
      <td>...</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1452384000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>West Ham</td>
      <td>Middlesbrough</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>N Swarbrick</td>
      <td>19</td>
      <td>...</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1452384000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Swansea</td>
      <td>Liverpool</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>M Oliver</td>
      <td>8</td>
      <td>...</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1452384000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Watford</td>
      <td>Bournemouth</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>M Dean</td>
      <td>17</td>
      <td>...</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1452384000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hull</td>
      <td>Chelsea</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>A Taylor</td>
      <td>8</td>
      <td>...</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1452384000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>Fulham</td>
      <td>Newcastle</td>
      <td>0</td>
      <td>4</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>K Friend</td>
      <td>16</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>339</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1575504000</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>Man United</td>
      <td>Cardiff</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>J Moss</td>
      <td>26</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>339</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1575504000</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>Southampton</td>
      <td>Huddersfield</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>L Probert</td>
      <td>10</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>339</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1575504000</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>Watford</td>
      <td>West Ham</td>
      <td>1</td>
      <td>4</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>C Kavanagh</td>
      <td>17</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>339</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1575504000</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>Tottenham</td>
      <td>Everton</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>A Marriner</td>
      <td>11</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>339</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1575504000</td>
    </tr>
  </tbody>
</table>
<p>1140 rows × 54 columns</p>
</div>



### The default fastai library split doesn't get it right


So the built in `cont_cat_split` categorizes continous variables with very few values as categorical but I manually declared the ones I wanted the model to treat as continuous. The next few lines show the split that `fastai` makes.

```python
dep_var = 'FTR'
cont_names, cat_names = cont_cat_split(df, dep_var=dep_var)
```

```python
cont_names
```



    ['HS',
     'AS',
     'HF',
     'AF',
     'LBH',
     'LBD',
     'LBA',
     'win3_streak',
     'home_win3_streak',
     'Week',
     'Day',
     'Dayofyear',
     'Elapsed']



```python
cat_names
```




    ['HomeTeam',
     'AwayTeam',
     'FTHG',
     'FTAG',
     'HTHG',
     'HTAG',
     'HTR',
     'Referee',
     'HST',
     'AST',
     'HC',
     'AC',
     'HY',
     'AY',
     'HR',
     'AR',
     'game_loc',
     'ht_goal_difference',
     'ft_goal_difference',
     'loss3_streak',
     'win5_streak',
     'loss5_streak',
     'away_win3_streak',
     'away_win5_streak',
     'away_loss3_streak',
     'away_loss5_streak',
     'home_win5_streak',
     'home_loss3_streak',
     'home_loss5_streak',
     'home_last4pts',
     'away_last4pts',
     'Year',
     'Month',
     'Dayofweek',
     'Is_month_end',
     'Is_month_start',
     'Is_quarter_end',
     'Is_quarter_start',
     'Is_year_end',
     'Is_year_start']



Here is the manual specification of variables and the dependent variable. We use `FTR - Full Time Result` which has 3 categories `['A', 'D', 'H']` as the variable to predict.

This also provides me with the ability to predict the probability for a win for each team.

I add all the basic tabular transforms to 
1. fill missing variables (which we don't have), 
2. categorize and encode categorical variables. 
3. Normalize continuous variables

I create the test,training and validation data sets and determine the learning rate. My validation data set is records between row indexes 900 and 1100 and the test set is the last 40 records in dataframe. 



```python
cont_names = ['HS', 'AS', 'HF', 'AF', 'LBH', 'LBD', 'LBA', 
              'FTHG', 'FTAG','HTHG','HTAG','HST', 'AST', 
              'HC', 'AC', 'HY', 'AY',
              'HR', 'AR', 'ht_goal_difference', 'ft_goal_difference',
              'loss3_streak', 'win5_streak', 'loss5_streak', 'away_win3_streak',
              'away_win5_streak','away_loss3_streak','away_loss5_streak',
              'home_win5_streak','home_loss3_streak','home_loss5_streak',
              'home_last4pts','away_last4pts','win3_streak', 'home_win3_streak',
              'Week', 'Day', 'Dayofyear', 'Elapsed']

cat_names = ['HomeTeam', 'AwayTeam', 'HTR', 'Referee']
dep_var='FTR'
procs = [FillMissing, Categorify, Normalize]
```

```python
test = (TabularList.from_df(df.iloc[1100:1140].copy(), 
                            path=MODEL_DATA_PATH, 
                            cat_names=cat_names, 
                            cont_names=cont_names,
                            procs=procs) )
                    



```

```python
data = (TabularList.from_df(df, path=MODEL_DATA_PATH, 
                           cat_names=cat_names,
                           cont_names=cont_names,
                           procs=procs)
                    .split_by_idx(list(range(900, 1100)))
                    .label_from_df(cols=dep_var)
                    .add_test(test)
                    .databunch(bs=64)
       )

data       
```


    TabularDataBunch;
    
    Train: LabelList (940 items)
    x: TabularList
    HomeTeam Sunderland; AwayTeam West Brom; HTR A; Referee S Attwell; LBH_na False; LBD_na False; LBA_na False; HS -1.1770; AS 1.1988; HF -1.0876; AF 0.5638; LBH -0.2038; LBD -0.6903; LBA -0.3624; FTHG -0.4246; FTAG -0.1694; HTHG -0.7922; HTAG 0.6260; HST -0.9981; AST 1.3226; HC 0.0907; AC 0.1299; HY -0.4927; AY 0.9943; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; ,HomeTeam West Ham; AwayTeam Middlesbrough; HTR D; Referee N Swarbrick; LBH_na False; LBD_na False; LBA_na False; HS 0.8697; AS -0.4445; HF 0.6956; AF 0.2811; LBH -0.3034; LBD -0.5185; LBA -0.3047; FTHG -0.4246; FTAG -0.1694; HTHG -0.7922; HTAG -0.7023; HST -0.9981; AST -0.3718; HC -0.5754; AC 0.1299; HY 0.3003; AY 0.9943; HR -0.2275; AR -0.2492; ht_goal_difference 0.0503; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; ,HomeTeam Swansea; AwayTeam Liverpool; HTR H; Referee M Oliver; LBH_na False; LBD_na False; LBA_na False; HS -1.0065; AS 1.4042; HF 0.1012; AF -0.5668; LBH 2.3357; LBD 0.6407; LBA -0.7804; FTHG -0.4246; FTAG 0.6731; HTHG 0.3993; HTAG -0.7023; HST -0.6346; AST 0.8990; HC -0.9085; AC 1.9573; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference -0.4110; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; ,HomeTeam Watford; AwayTeam Bournemouth; HTR A; Referee M Dean; LBH_na False; LBD_na False; LBA_na False; HS 0.5285; AS -0.0337; HF 1.8844; AF 0.2811; LBH -0.2138; LBD -0.6473; LBA -0.3624; FTHG 0.3299; FTAG 0.6731; HTHG -0.7922; HTAG 0.6260; HST 0.8194; AST -0.7954; HC -0.5754; AC 0.1299; HY 1.0933; AY 1.7719; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; ,HomeTeam Hull; AwayTeam Chelsea; HTR D; Referee A Taylor; LBH_na False; LBD_na False; LBA_na False; HS -1.0065; AS 2.2259; HF 0.6956; AF 1.1290; LBH 2.0867; LBD 0.3402; LBA -0.7666; FTHG -1.1792; FTAG 0.6731; HTHG -0.7922; HTAG -0.7023; HST -0.6346; AST 2.1698; HC -0.2424; AC 0.8608; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference 0.0503; ft_goal_difference -0.9316; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; 
    y: CategoryList
    D,D,A,D,A
    Path: /home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data;
    
    Valid: LabelList (200 items)
    x: TabularList
    HomeTeam Watford; AwayTeam Brighton; HTR H; Referee J Moss; LBH_na True; LBD_na True; LBA_na True; HS 0.8697; AS -1.0607; HF -0.1960; AF 1.4117; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 0.3299; FTAG -1.0119; HTHG 0.3993; HTAG -0.7023; HST 0.0924; AST -1.6426; HC 0.7569; AC -0.9666; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference 1.1509; loss3_streak -0.2023; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.2023; home_loss5_streak -0.2825; home_last4pts 0.2870; away_last4pts -0.8329; win3_streak -0.1580; home_win3_streak -0.1580; Week 0.9939; Day -0.8237; Dayofyear 0.9797; Elapsed 1.3691; ,HomeTeam Wolves; AwayTeam Everton; HTR D; Referee C Pawson; LBH_na True; LBD_na True; LBA_na True; HS -0.4948; AS -1.0607; HF -0.7904; AF -1.1320; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 0.3299; FTAG 0.6731; HTHG 0.3993; HTAG 0.6260; HST -0.2711; AST 0.4754; HC -0.9085; AC 0.4954; HY -1.2857; AY -0.5609; HR -0.2275; AR 4.0092; ht_goal_difference 0.0503; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.2483; away_last4pts -0.2741; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.9939; Day -0.8237; Dayofyear 0.9797; Elapsed 1.3691; ,HomeTeam Arsenal; AwayTeam Wolves; HTR A; Referee S Attwell; LBH_na True; LBD_na True; LBA_na True; HS -0.6654; AS 0.1718; HF -0.4932; AF 1.4117; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG -0.4246; FTAG -0.1694; HTHG -0.7922; HTAG 0.6260; HST -0.6346; AST 0.4754; HC 1.7561; AC -0.9666; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak 0.2768; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak 0.2768; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts 4.5688; away_last4pts -0.2741; win3_streak 0.5094; home_win3_streak 0.5094; Week 0.9939; Day -0.4926; Dayofyear 1.0059; Elapsed 1.3794; ,HomeTeam Man City; AwayTeam Man United; HTR H; Referee A Taylor; LBH_na True; LBD_na True; LBA_na True; HS 0.5285; AS -1.0607; HF 0.3984; AF 0.2811; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 1.0845; FTAG -0.1694; HTHG 0.3993; HTAG -0.7023; HST 0.0924; AST -1.2190; HC -0.2424; AC -1.3321; HY -0.4927; AY -0.5609; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference 1.1509; loss3_streak -0.6962; win5_streak 0.8374; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak 0.8374; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.2741; win3_streak 1.5104; home_win3_streak 1.5104; Week 0.9939; Day -0.4926; Dayofyear 1.0059; Elapsed 1.3794; ,HomeTeam Liverpool; AwayTeam Fulham; HTR H; Referee P Tierney; LBH_na True; LBD_na True; LBA_na True; HS 1.0402; AS -0.6499; HF 0.1012; AF -0.5668; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 0.3299; FTAG -1.0119; HTHG 0.3993; HTAG -0.7023; HST 0.8194; AST -0.3718; HC 0.0907; AC -0.6011; HY -0.4927; AY -0.5609; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference -0.9316; loss3_streak 1.2794; win5_streak -0.2839; loss5_streak 0.9943; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak 1.2794; home_loss5_streak 0.9943; home_last4pts -0.2483; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.9939; Day -0.4926; Dayofyear 1.0059; Elapsed 1.3794; 
    y: CategoryList
    H,D,D,H,H
    Path: /home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data;
    
    Test: LabelList (40 items)
    x: TabularList
    HomeTeam Everton; AwayTeam Arsenal; HTR H; Referee K Friend; LBH_na True; LBD_na True; LBA_na True; HS 1.5519; AS -0.8553; HF -0.7904; AF -0.5668; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG -0.4246; FTAG -1.0119; HTHG 0.3993; HTAG -0.7023; HST 0.4559; AST -0.7954; HC 1.0899; AC 0.4954; HY -0.4927; AY 1.7719; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference -0.4110; loss3_streak -0.2023; win5_streak 0.2768; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak 0.2768; home_loss3_streak -0.2023; home_loss5_streak -0.2825; home_last4pts -0.2483; away_last4pts -0.2741; win3_streak 1.1767; home_win3_streak 1.1767; Week -0.1128; Day -1.2652; Dayofyear -0.1309; Elapsed 2.1870; ,HomeTeam Chelsea; AwayTeam West Ham; HTR H; Referee C Kavanagh; LBH_na True; LBD_na True; LBA_na True; HS 0.3580; AS -0.4445; HF -0.7904; AF -1.1320; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 0.3299; FTAG -1.0119; HTHG 0.3993; HTAG -0.7023; HST 0.8194; AST -0.7954; HC 0.4238; AC -0.2356; HY 0.3003; AY -0.5609; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference -0.9316; loss3_streak -0.2023; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.2023; home_loss5_streak -0.2825; home_last4pts 0.2870; away_last4pts -0.2741; win3_streak -0.1580; home_win3_streak -0.1580; Week 0.1331; Day -1.2652; Dayofyear 0.1402; Elapsed 2.2935; ,HomeTeam Huddersfield; AwayTeam Arsenal; HTR A; Referee J Moss; LBH_na True; LBD_na True; LBA_na True; HS 0.1874; AS -0.4445; HF 1.8844; AF 0.2811; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG -0.4246; FTAG 0.6731; HTHG -0.7922; HTAG 1.9542; HST 0.4559; AST 0.0518; HC -0.2424; AC -1.6976; HY 1.0933; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference -1.6374; ft_goal_difference -0.4110; loss3_streak 5.7245; win5_streak -0.2839; loss5_streak 8.6552; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak 5.7245; home_loss5_streak 8.6552; home_last4pts -0.2483; away_last4pts -0.2741; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.4405; Day -1.4859; Dayofyear 0.3938; Elapsed 2.3932; ,HomeTeam Brighton; AwayTeam Burnley; HTR A; Referee S Attwell; LBH_na True; LBD_na True; LBA_na True; HS 0.3580; AS -0.4445; HF -0.7904; AF -1.1320; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG -0.4246; FTAG 1.5155; HTHG -0.7922; HTAG 0.6260; HST 0.4559; AST 0.4754; HC 1.0899; AC -0.6011; HY -0.4927; AY -0.5609; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference 1.1509; loss3_streak 2.2672; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak 2.2672; home_loss5_streak -0.2825; home_last4pts 0.8222; away_last4pts -0.2741; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.4405; Day -1.4859; Dayofyear 0.3938; Elapsed 2.3932; ,HomeTeam Liverpool; AwayTeam Bournemouth; HTR H; Referee A Taylor; LBH_na True; LBD_na True; LBA_na True; HS 1.0402; AS 0.1718; HF 0.9928; AF -1.4147; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 1.0845; FTAG -1.0119; HTHG 1.5908; HTAG -0.7023; HST 1.5465; AST -0.7954; HC 0.7569; AC 0.1299; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference -1.6374; ft_goal_difference -1.4522; loss3_streak -0.2023; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.2023; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.2741; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.4405; Day -1.4859; Dayofyear 0.3938; Elapsed 2.3932; 
    y: EmptyLabelList
    ,,,,
    Path: /home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data



```python
data.show_batch(rows=10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>HTR</th>
      <th>Referee</th>
      <th>LBH_na</th>
      <th>LBD_na</th>
      <th>LBA_na</th>
      <th>HS</th>
      <th>AS</th>
      <th>HF</th>
      <th>AF</th>
      <th>LBH</th>
      <th>LBD</th>
      <th>LBA</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HST</th>
      <th>AST</th>
      <th>HC</th>
      <th>AC</th>
      <th>HY</th>
      <th>AY</th>
      <th>HR</th>
      <th>AR</th>
      <th>ht_goal_difference</th>
      <th>ft_goal_difference</th>
      <th>loss3_streak</th>
      <th>win5_streak</th>
      <th>loss5_streak</th>
      <th>away_win3_streak</th>
      <th>away_win5_streak</th>
      <th>away_loss3_streak</th>
      <th>away_loss5_streak</th>
      <th>home_win5_streak</th>
      <th>home_loss3_streak</th>
      <th>home_loss5_streak</th>
      <th>home_last4pts</th>
      <th>away_last4pts</th>
      <th>win3_streak</th>
      <th>home_win3_streak</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofyear</th>
      <th>Elapsed</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Everton</td>
      <td>West Brom</td>
      <td>A</td>
      <td>S Attwell</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-0.8359</td>
      <td>0.7880</td>
      <td>1.2900</td>
      <td>-0.2841</td>
      <td>-0.3532</td>
      <td>-0.7761</td>
      <td>-0.2585</td>
      <td>-0.4246</td>
      <td>-0.1694</td>
      <td>-0.7922</td>
      <td>0.6260</td>
      <td>-0.2711</td>
      <td>0.4754</td>
      <td>-1.9077</td>
      <td>-0.6011</td>
      <td>0.3003</td>
      <td>-0.5609</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>-0.7936</td>
      <td>0.1097</td>
      <td>0.2916</td>
      <td>-0.2839</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2839</td>
      <td>0.2916</td>
      <td>-0.2825</td>
      <td>-0.2483</td>
      <td>0.2848</td>
      <td>0.1757</td>
      <td>0.1757</td>
      <td>-1.5885</td>
      <td>0.5008</td>
      <td>-1.5739</td>
      <td>0.3656</td>
      <td>D</td>
    </tr>
    <tr>
      <td>Arsenal</td>
      <td>Leicester</td>
      <td>D</td>
      <td>C Kavanagh</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>0.8697</td>
      <td>-0.6499</td>
      <td>-0.1960</td>
      <td>-0.2841</td>
      <td>-0.3034</td>
      <td>-0.3468</td>
      <td>-0.3278</td>
      <td>1.0845</td>
      <td>-0.1694</td>
      <td>0.3993</td>
      <td>0.6260</td>
      <td>0.4559</td>
      <td>-0.7954</td>
      <td>0.0907</td>
      <td>-0.2356</td>
      <td>0.3003</td>
      <td>0.2167</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>0.0503</td>
      <td>1.1509</td>
      <td>-0.6962</td>
      <td>0.2768</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2768</td>
      <td>-0.6962</td>
      <td>-0.2825</td>
      <td>5.6393</td>
      <td>-0.2741</td>
      <td>0.5094</td>
      <td>0.5094</td>
      <td>0.8710</td>
      <td>0.7215</td>
      <td>0.8310</td>
      <td>1.3107</td>
      <td>H</td>
    </tr>
    <tr>
      <td>Man United</td>
      <td>Arsenal</td>
      <td>D</td>
      <td>A Marriner</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-0.3242</td>
      <td>-1.2661</td>
      <td>0.9928</td>
      <td>-0.0015</td>
      <td>-0.1540</td>
      <td>-0.6044</td>
      <td>-0.4202</td>
      <td>-0.4246</td>
      <td>-0.1694</td>
      <td>-0.7922</td>
      <td>-0.7023</td>
      <td>0.0924</td>
      <td>-1.2190</td>
      <td>1.4230</td>
      <td>-0.2356</td>
      <td>1.0933</td>
      <td>0.9943</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>0.0503</td>
      <td>0.1097</td>
      <td>-0.6962</td>
      <td>-0.2839</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2839</td>
      <td>-0.6962</td>
      <td>-0.2825</td>
      <td>0.2870</td>
      <td>4.7553</td>
      <td>0.1757</td>
      <td>0.1757</td>
      <td>1.0554</td>
      <td>0.3904</td>
      <td>1.0846</td>
      <td>-1.1017</td>
      <td>D</td>
    </tr>
    <tr>
      <td>West Brom</td>
      <td>Newcastle</td>
      <td>H</td>
      <td>L Probert</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1.0065</td>
      <td>0.5826</td>
      <td>0.6956</td>
      <td>0.2811</td>
      <td>-0.2287</td>
      <td>-0.7761</td>
      <td>-0.3278</td>
      <td>0.3299</td>
      <td>0.6731</td>
      <td>0.3993</td>
      <td>-0.7023</td>
      <td>-0.2711</td>
      <td>0.0518</td>
      <td>-1.5746</td>
      <td>0.4954</td>
      <td>-1.2857</td>
      <td>-1.3384</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>0.8941</td>
      <td>0.1097</td>
      <td>-0.2023</td>
      <td>-0.2839</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2839</td>
      <td>-0.2023</td>
      <td>-0.2825</td>
      <td>-0.2483</td>
      <td>-0.8329</td>
      <td>-0.4916</td>
      <td>-0.4916</td>
      <td>1.1784</td>
      <td>1.3838</td>
      <td>1.1546</td>
      <td>0.1835</td>
      <td>D</td>
    </tr>
    <tr>
      <td>Man City</td>
      <td>Crystal Palace</td>
      <td>H</td>
      <td>M Oliver</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2.0635</td>
      <td>-1.2661</td>
      <td>0.3984</td>
      <td>-0.8494</td>
      <td>-0.7615</td>
      <td>1.4994</td>
      <td>1.4274</td>
      <td>2.5935</td>
      <td>-1.0119</td>
      <td>0.3993</td>
      <td>-0.7023</td>
      <td>2.6370</td>
      <td>-0.7954</td>
      <td>1.0899</td>
      <td>-0.6011</td>
      <td>-0.4927</td>
      <td>0.9943</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>0.8941</td>
      <td>2.7128</td>
      <td>-0.6962</td>
      <td>1.9588</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.9588</td>
      <td>-0.6962</td>
      <td>-0.2825</td>
      <td>-0.7835</td>
      <td>-0.8329</td>
      <td>2.8450</td>
      <td>2.8450</td>
      <td>-0.3588</td>
      <td>-1.1548</td>
      <td>-0.3846</td>
      <td>-0.4213</td>
      <td>H</td>
    </tr>
    <tr>
      <td>Huddersfield</td>
      <td>Everton</td>
      <td>A</td>
      <td>L Probert</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-0.8359</td>
      <td>-0.8553</td>
      <td>-0.4932</td>
      <td>-0.8494</td>
      <td>-0.1540</td>
      <td>-0.8620</td>
      <td>-0.4202</td>
      <td>-1.1792</td>
      <td>0.6731</td>
      <td>-0.7922</td>
      <td>0.6260</td>
      <td>-0.9981</td>
      <td>0.4754</td>
      <td>-0.5754</td>
      <td>-0.6011</td>
      <td>-0.4927</td>
      <td>-1.3384</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>0.8941</td>
      <td>1.1509</td>
      <td>0.7855</td>
      <td>-0.2839</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2839</td>
      <td>0.7855</td>
      <td>-0.2825</td>
      <td>-0.7835</td>
      <td>-0.2741</td>
      <td>0.1757</td>
      <td>0.1757</td>
      <td>-0.7277</td>
      <td>1.3838</td>
      <td>-0.7169</td>
      <td>0.7024</td>
      <td>A</td>
    </tr>
    <tr>
      <td>Arsenal</td>
      <td>West Ham</td>
      <td>D</td>
      <td>M Atkinson</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.2108</td>
      <td>-0.8553</td>
      <td>0.1012</td>
      <td>-1.6973</td>
      <td>-0.7217</td>
      <td>0.8554</td>
      <td>0.6191</td>
      <td>1.0845</td>
      <td>-1.0119</td>
      <td>-0.7922</td>
      <td>-0.7023</td>
      <td>1.1829</td>
      <td>-0.7954</td>
      <td>-0.2424</td>
      <td>-1.6976</td>
      <td>0.3003</td>
      <td>0.2167</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>0.0503</td>
      <td>-1.4522</td>
      <td>0.7855</td>
      <td>-0.2839</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2839</td>
      <td>0.7855</td>
      <td>-0.2825</td>
      <td>2.4279</td>
      <td>0.8436</td>
      <td>-0.1580</td>
      <td>-0.1580</td>
      <td>-0.6662</td>
      <td>-1.2652</td>
      <td>-0.6644</td>
      <td>-0.5313</td>
      <td>H</td>
    </tr>
    <tr>
      <td>Middlesbrough</td>
      <td>Burnley</td>
      <td>D</td>
      <td>M Atkinson</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-0.3242</td>
      <td>-0.8553</td>
      <td>-0.4932</td>
      <td>0.2811</td>
      <td>-0.2138</td>
      <td>-0.8620</td>
      <td>-0.3624</td>
      <td>-1.1792</td>
      <td>-1.0119</td>
      <td>-0.7922</td>
      <td>-0.7023</td>
      <td>0.0924</td>
      <td>-0.7954</td>
      <td>-0.5754</td>
      <td>-0.6011</td>
      <td>-1.2857</td>
      <td>0.9943</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>0.0503</td>
      <td>0.1097</td>
      <td>0.2916</td>
      <td>-0.2839</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2839</td>
      <td>0.2916</td>
      <td>-0.2825</td>
      <td>-0.2483</td>
      <td>-0.2741</td>
      <td>-0.4916</td>
      <td>-0.4916</td>
      <td>0.1331</td>
      <td>-1.2652</td>
      <td>0.1402</td>
      <td>-0.2151</td>
      <td>D</td>
    </tr>
    <tr>
      <td>Burnley</td>
      <td>Stoke</td>
      <td>D</td>
      <td>K Friend</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1.1770</td>
      <td>0.7880</td>
      <td>-0.4932</td>
      <td>1.4117</td>
      <td>-0.2038</td>
      <td>-0.7761</td>
      <td>-0.3509</td>
      <td>-0.4246</td>
      <td>-1.0119</td>
      <td>-0.7922</td>
      <td>-0.7023</td>
      <td>-0.6346</td>
      <td>-0.7954</td>
      <td>-0.9085</td>
      <td>0.1299</td>
      <td>1.0933</td>
      <td>0.2167</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>0.0503</td>
      <td>-0.4110</td>
      <td>-0.2023</td>
      <td>-0.2839</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2839</td>
      <td>-0.2023</td>
      <td>-0.2825</td>
      <td>0.2870</td>
      <td>-0.8329</td>
      <td>-0.1580</td>
      <td>-0.1580</td>
      <td>-0.9122</td>
      <td>-1.2652</td>
      <td>-0.9267</td>
      <td>-0.6344</td>
      <td>H</td>
    </tr>
    <tr>
      <td>Southampton</td>
      <td>Chelsea</td>
      <td>A</td>
      <td>C Pawson</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>0.1874</td>
      <td>2.0204</td>
      <td>0.6956</td>
      <td>-0.0015</td>
      <td>-0.3034</td>
      <td>-0.3468</td>
      <td>-0.3278</td>
      <td>-1.1792</td>
      <td>1.5155</td>
      <td>-0.7922</td>
      <td>0.6260</td>
      <td>0.4559</td>
      <td>0.8990</td>
      <td>-0.5754</td>
      <td>2.6883</td>
      <td>3.4724</td>
      <td>-1.3384</td>
      <td>-0.2275</td>
      <td>-0.2492</td>
      <td>-0.7936</td>
      <td>-1.4522</td>
      <td>-0.2023</td>
      <td>-0.2839</td>
      <td>-0.2825</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2839</td>
      <td>-0.2023</td>
      <td>-0.2825</td>
      <td>-0.2483</td>
      <td>-0.8329</td>
      <td>-0.4916</td>
      <td>-0.4916</td>
      <td>-0.0513</td>
      <td>-0.6029</td>
      <td>-0.0785</td>
      <td>0.9533</td>
      <td>A</td>
    </tr>
  </tbody>
</table>


```python
learn = None
learn = tabular_learner(data, layers=[400, 100], metrics=[accuracy] )
```

```python
learn.lr_find()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>      
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.187053</td>
      <td>na</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.181402</td>
      <td>na</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.148650</td>
      <td>na</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.991824</td>
      <td>na</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.786385</td>
      <td>na</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.711735</td>
      <td>na</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


```python
LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
```

```python
learn.recorder.plot(suggestion=True)
```

    Min numerical gradient: 3.63E-03
    Min loss divided by 10: 2.09E-02



![png](/images/EPL_predictions_train_files/output_25_1.png)


I have a pretty decent accuracy right off the bat. I try once more to check if there can be any possible improvements on the accuracy.

```python
learn.fit_one_cycle(2, 5e-02)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.643659</td>
      <td>0.732382</td>
      <td>0.800000</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.449729</td>
      <td>0.145192</td>
      <td>0.955000</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


```python
MODEL_PATH = Path('/home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/models')
```

```python
learn.save(MODEL_PATH/'model')
```

```python
learn.lr_find()
learn.recorder.plot(suggestion=True, skip_start=2, skip_end=2)
```
   

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.167630</td>
      <td>#na#</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.171821</td>
      <td>#na#</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.162816</td>
      <td>#na#</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151836</td>
      <td>#na#</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.147282</td>
      <td>#na#</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.317775</td>
      <td>#na#</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


```python
  LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
  Min numerical gradient: 3.02E-07
  Min loss divided by 10: 8.32E-03
```


![png](/images/EPL_predictions_train_files/output_30_2.png)


```python
learn.fit_one_cycle(1, 4e-07)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.167380</td>
      <td>0.154288</td>
      <td>0.950000</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


No real improvement and the accuracy has marginally slipped. It would make sense, not to save this model and use our last saved version. 
Either ways the drift is marginal. 

```python
learn.save(MODEL_PATH/'model2')
```

## Predict

Now the easiest way is to plug in the test data loader into the learner and give it a go.


```python
test1 = (TabularList.from_df(df.iloc[1100:1140].copy(), 
                            path=MODEL_DATA_PATH, 
                            cat_names=cat_names, 
                            cont_names=cont_names,
                            procs=procs)
                     .split_none()
                     .label_from_df(cols=dep_var)
        )   

test1.valid = test1.train
test1 = test1.databunch(bs=16)
valid_dl = learn.data.valid_dl
learn.data.valid_dl = test1.valid_dl
```
<br>

A few helper functions to display the batch preds in a easy interpretable way

```python
def get_pred_class(pred_idx): return data.y.classes[pred_idx]
def win_prob(pred, home_tm, away_tm):
    preds = pred.data.numpy() * 100
    return ((home_tm, preds[2]), ('draw', preds[1]), (away_tm, preds[0]))


p = {
    'home_team': [],
    'away_team': [],
    'real_winner': [],
    'predicted': [],
    'prob': []
}
for idx, row in df.iloc[1100:].iterrows():
    cat, preds, y = learn.predict(row)
    p['home_team'].append(row.HomeTeam)
    p['away_team'].append(row.AwayTeam)
    p['real_winner'].append(row.FTR)
    p['predicted'].append(get_pred_class(cat.data))
    p['prob'].append(win_prob(y, row.HomeTeam, row.AwayTeam))
    print(f'{row.HomeTeam} v {row.AwayTeam} | winner: {row.FTR} | predicted: {get_pred_class(cat.data)} | prob: {win_prob(y, row.HomeTeam, row.AwayTeam)}')
     
    
pred_df = pd.DataFrame(p, columns=['home_team', 'away_team', 'real_winner', 'predicted', 'prob'])
pred_df
```

In case we decide to do it as a batch this would be the way to go. 

```python
preds, y = learn.get_preds(DatasetType.Valid)
preds, y
```


```
[tensor([[2.0051e-08, 1.0250e-06, 1.0000e+00],
         [6.2912e-03, 8.2377e-01, 1.6994e-01],
         [4.3451e-03, 9.5819e-01, 3.7469e-02],
         [6.0256e-08, 1.9831e-06, 1.0000e+00],
         [1.0940e-06, 1.0939e-04, 9.9989e-01],
         [1.5659e-03, 9.6968e-01, 2.8755e-02],
         [1.0000e+00, 1.0993e-07, 1.0714e-11],
         [5.0318e-05, 1.1147e-02, 9.8880e-01],
         [1.5541e-03, 9.4054e-01, 5.7907e-02],
         [3.1620e-08, 8.8461e-07, 1.0000e+00],
         [1.2153e-03, 9.4577e-01, 5.3010e-02],
         [1.0000e+00, 2.4524e-11, 6.1091e-16],
         [1.0271e-05, 2.1882e-03, 9.9780e-01],
         [9.8494e-01, 1.5033e-02, 2.7221e-05],
         [1.0000e+00, 3.4486e-06, 5.1989e-10],
         [9.9140e-01, 8.5899e-03, 1.1520e-05],
         [7.7035e-05, 2.0601e-02, 9.7932e-01],
         [7.0603e-11, 2.4852e-10, 1.0000e+00],
         [2.6422e-03, 9.7562e-01, 2.1737e-02],
         [9.9970e-01, 3.0337e-04, 3.5320e-07],
         [9.8659e-01, 1.3391e-02, 1.5097e-05],
         [1.7928e-05, 2.2702e-03, 9.9771e-01],
         [9.9998e-01, 2.2737e-05, 4.3796e-09],
         [1.2433e-05, 2.0520e-03, 9.9794e-01],
         [5.3454e-08, 2.8971e-06, 1.0000e+00],
         [1.5739e-04, 5.9541e-02, 9.4030e-01],
         [3.2945e-07, 7.6986e-06, 9.9999e-01],
         [4.0183e-04, 1.6876e-01, 8.3084e-01],
         [9.8598e-01, 1.4003e-02, 2.1703e-05],
         [7.3439e-06, 8.1140e-04, 9.9918e-01],
         [9.9995e-01, 5.1998e-05, 3.8103e-08],
         [9.9989e-01, 1.0886e-04, 3.7874e-08],
         [8.3157e-01, 1.6813e-01, 2.9190e-04],
         [9.0401e-01, 9.5341e-02, 6.4794e-04],
         [1.4144e-03, 9.7152e-01, 2.7064e-02],
         [1.0000e+00, 3.4484e-12, 5.4901e-17],
         [9.9999e-01, 6.3331e-06, 1.9858e-09],
         [4.7619e-08, 2.4740e-06, 1.0000e+00],
         [8.8786e-09, 2.5347e-07, 1.0000e+00],
         [1.0000e+00, 1.3758e-11, 6.6039e-16],
         [7.3305e-05, 8.4544e-03, 9.9147e-01],
         [4.3369e-03, 9.8152e-01, 1.4147e-02],
         [9.8093e-01, 1.9040e-02, 3.1459e-05],
         [2.5006e-15, 3.7738e-16, 1.0000e+00],
         [5.6457e-07, 4.0223e-05, 9.9996e-01],
         [3.0686e-03, 9.7244e-01, 2.4490e-02],
         [1.5133e-10, 7.4305e-10, 1.0000e+00],
         [7.1577e-03, 9.5490e-01, 3.7946e-02],
         [1.0000e+00, 1.2809e-11, 2.5820e-16],
         [7.4365e-01, 2.5540e-01, 9.4438e-04],
         [9.0667e-06, 1.5400e-03, 9.9845e-01],
         [7.3865e-01, 2.6089e-01, 4.5760e-04],
         [9.9992e-01, 7.5468e-05, 4.4912e-08],
         [2.9930e-04, 1.4120e-01, 8.5850e-01],
         [7.0433e-03, 9.7617e-01, 1.6790e-02],
         [6.3485e-10, 1.9191e-09, 1.0000e+00],
         [1.7239e-08, 9.1305e-07, 1.0000e+00],
         [9.4074e-01, 5.9174e-02, 8.4152e-05],
         [9.9999e-01, 1.0707e-05, 2.2708e-09],
         [8.5536e-09, 1.2038e-07, 1.0000e+00],
         [9.2613e-01, 7.3789e-02, 7.8120e-05],
         [1.0000e+00, 4.0934e-07, 7.0205e-11],
         [5.8857e-09, 1.1071e-07, 1.0000e+00],
         [2.2252e-07, 1.5476e-05, 9.9998e-01],
         [3.5767e-01, 6.3866e-01, 3.6676e-03],
         [8.4564e-08, 5.6851e-06, 9.9999e-01],
         [8.1584e-09, 7.1270e-08, 1.0000e+00],
         [2.8274e-04, 1.4001e-01, 8.5971e-01],
         [3.4719e-08, 1.6765e-06, 1.0000e+00],
         [9.1471e-06, 1.3780e-03, 9.9861e-01],
         [1.9915e-09, 3.3895e-08, 1.0000e+00],
         [2.8075e-03, 3.4405e-01, 6.5315e-01],
         [3.3313e-08, 1.1647e-06, 1.0000e+00],
         [1.2274e-03, 9.6908e-01, 2.9689e-02],
         [1.5338e-04, 4.1555e-02, 9.5829e-01],
         [5.0213e-01, 4.9529e-01, 2.5754e-03],
         [1.0000e+00, 2.1775e-08, 7.6614e-13],
         [1.3922e-06, 1.5733e-04, 9.9984e-01],
         [1.5799e-09, 3.2996e-08, 1.0000e+00],
         [3.3690e-04, 1.3104e-01, 8.6862e-01],
         [5.3880e-05, 1.3839e-02, 9.8611e-01],
         [3.0079e-03, 9.6067e-01, 3.6323e-02],
         [5.4620e-01, 4.5295e-01, 8.5130e-04],
         [1.9431e-03, 9.7704e-01, 2.1021e-02],
         [2.6660e-03, 8.7079e-01, 1.2655e-01],
         [2.8888e-04, 6.4555e-02, 9.3516e-01],
         [4.2101e-13, 6.6452e-13, 1.0000e+00],
         [9.9990e-01, 9.7159e-05, 3.1167e-08],
         [1.1342e-03, 9.5486e-01, 4.4001e-02],
         [8.5905e-03, 9.2275e-01, 6.8660e-02],
         [9.9999e-01, 9.7550e-06, 1.1301e-09],
         [9.3266e-01, 6.7230e-02, 1.0897e-04],
         [3.9115e-03, 7.3482e-01, 2.6127e-01],
         [6.0041e-10, 5.4048e-09, 1.0000e+00],
         [1.1284e-03, 9.2295e-01, 7.5925e-02],
         [1.3124e-06, 1.7020e-04, 9.9983e-01],
         [9.9998e-01, 1.5319e-05, 4.0238e-09],
         [1.9504e-07, 1.7338e-05, 9.9998e-01],
         [4.5122e-03, 9.7400e-01, 2.1491e-02],
         [7.5449e-05, 1.0782e-02, 9.8914e-01],
         [8.7481e-07, 9.2914e-05, 9.9991e-01],
         [5.2036e-04, 3.3832e-01, 6.6116e-01],
         [6.5914e-01, 3.3820e-01, 2.6523e-03],
         [2.0867e-08, 1.1933e-06, 1.0000e+00],
         [9.9997e-01, 2.7080e-05, 7.9754e-09],
         [3.0707e-03, 9.2773e-01, 6.9200e-02],
         [2.9212e-04, 1.2539e-01, 8.7432e-01],
         [1.9386e-05, 3.9083e-03, 9.9607e-01],
         [1.1301e-10, 9.9393e-10, 1.0000e+00],
         [1.0000e+00, 3.8809e-10, 1.0466e-14],
         [7.6345e-08, 4.7698e-06, 1.0000e+00],
         [1.0000e+00, 7.1187e-07, 6.3179e-11],
         [1.2979e-03, 9.7234e-01, 2.6362e-02],
         [4.3319e-04, 1.0212e-01, 8.9744e-01],
         [3.7007e-05, 9.9883e-03, 9.8997e-01],
         [2.8007e-07, 1.6174e-05, 9.9998e-01],
         [1.1171e-03, 9.0397e-01, 9.4909e-02],
         [7.3747e-04, 5.1015e-01, 4.8912e-01],
         [3.9470e-05, 9.8397e-03, 9.9012e-01],
         [4.7756e-07, 4.6670e-05, 9.9995e-01],
         [1.0000e+00, 6.7053e-08, 2.9022e-12],
         [1.2244e-12, 1.1498e-12, 1.0000e+00],
         [9.9984e-01, 1.6453e-04, 5.1466e-08],
         [5.2102e-07, 2.1912e-05, 9.9998e-01],
         [5.6224e-12, 1.3142e-11, 1.0000e+00],
         [2.3449e-06, 2.6625e-04, 9.9973e-01],
         [8.1402e-05, 2.8432e-02, 9.7149e-01],
         [1.6441e-04, 2.3915e-02, 9.7592e-01],
         [6.9318e-07, 1.2300e-05, 9.9999e-01],
         [5.2000e-01, 4.7893e-01, 1.0677e-03],
         [9.8057e-01, 1.9402e-02, 2.4482e-05],
         [5.2500e-05, 1.9718e-02, 9.8023e-01],
         [1.0736e-03, 9.5955e-01, 3.9374e-02],
         [1.2067e-08, 3.0626e-07, 1.0000e+00],
         [4.0929e-06, 1.1569e-04, 9.9988e-01],
         [3.1287e-07, 1.0990e-05, 9.9999e-01],
         [5.7103e-07, 5.2095e-05, 9.9995e-01],
         [9.2079e-02, 9.0240e-01, 5.5202e-03],
         [8.3045e-04, 4.2220e-01, 5.7697e-01],
         [8.8230e-01, 1.1754e-01, 1.5416e-04],
         [9.9135e-01, 8.6396e-03, 6.5196e-06],
         [7.5588e-09, 2.6280e-07, 1.0000e+00],
         [9.9999e-01, 6.0723e-06, 9.1039e-10],
         [9.9997e-01, 2.5192e-05, 1.1510e-08],
         [2.3995e-05, 5.7824e-03, 9.9419e-01],
         [9.9305e-06, 1.5920e-03, 9.9840e-01],
         [4.9029e-01, 5.0871e-01, 1.0006e-03],
         [2.9363e-07, 2.4547e-05, 9.9998e-01],
         [9.0843e-08, 6.5620e-06, 9.9999e-01],
         [9.6442e-01, 3.5540e-02, 4.0140e-05],
         [1.1668e-04, 1.6933e-02, 9.8295e-01],
         [1.1952e-02, 9.7388e-01, 1.4164e-02],
         [9.8031e-01, 1.9639e-02, 5.2756e-05],
         [9.7492e-01, 2.5030e-02, 4.6443e-05],
         [8.9838e-11, 6.5796e-10, 1.0000e+00],
         [2.2565e-04, 1.1868e-01, 8.8109e-01],
         [7.1050e-05, 1.2195e-02, 9.8773e-01],
         [5.7947e-06, 1.1273e-03, 9.9887e-01],
         [1.0054e-11, 2.2505e-11, 1.0000e+00],
         [2.9132e-05, 6.1546e-03, 9.9382e-01],
         [1.3490e-07, 1.3143e-05, 9.9999e-01],
         [1.0000e+00, 6.8594e-15, 1.0284e-20],
         [4.9874e-07, 5.1488e-05, 9.9995e-01],
         [9.8716e-07, 1.9659e-05, 9.9998e-01],
         [9.9941e-01, 5.8842e-04, 2.2202e-06],
         [9.9631e-01, 3.6835e-03, 3.6951e-06],
         [9.9993e-01, 6.7542e-05, 9.3057e-09],
         [8.6322e-01, 1.3657e-01, 2.0451e-04],
         [3.5355e-02, 9.5404e-01, 1.0609e-02],
         [1.7677e-01, 8.2085e-01, 2.3774e-03],
         [1.1840e-08, 4.1082e-07, 1.0000e+00],
         [5.5252e-03, 9.7984e-01, 1.4639e-02],
         [6.0276e-05, 8.1371e-03, 9.9180e-01],
         [9.9995e-01, 5.3671e-05, 9.2738e-09],
         [6.4632e-13, 1.4714e-12, 1.0000e+00],
         [9.7167e-01, 2.8241e-02, 9.3613e-05],
         [1.4928e-02, 9.6849e-01, 1.6584e-02],
         [9.5643e-03, 9.7970e-01, 1.0735e-02],
         [2.6950e-04, 5.7772e-02, 9.4196e-01],
         [9.9202e-01, 7.9378e-03, 4.6166e-05],
         [2.2270e-06, 1.9584e-04, 9.9980e-01],
         [5.9206e-10, 4.0381e-09, 1.0000e+00],
         [2.8142e-03, 9.7825e-01, 1.8936e-02],
         [1.0407e-03, 7.7026e-01, 2.2870e-01],
         [9.7635e-01, 2.3613e-02, 3.3295e-05],
         [1.9975e-02, 9.6479e-01, 1.5238e-02],
         [1.1929e-02, 9.7989e-01, 8.1767e-03],
         [3.7624e-01, 6.1844e-01, 5.3203e-03],
         [9.7628e-01, 2.3694e-02, 2.9872e-05],
         [3.1192e-08, 6.3648e-07, 1.0000e+00],
         [3.6321e-03, 9.7443e-01, 2.1935e-02],
         [9.9982e-01, 1.7953e-04, 5.4221e-08],
         [7.3868e-10, 7.4133e-09, 1.0000e+00],
         [9.9784e-03, 9.7878e-01, 1.1241e-02],
         [4.1847e-03, 9.7650e-01, 1.9312e-02],
         [9.9920e-01, 7.9761e-04, 5.0813e-06],
         [1.0000e+00, 5.6043e-09, 1.7189e-13],
         [9.9999e-01, 6.8447e-06, 9.3081e-10],
         [8.9872e-01, 1.0116e-01, 1.2437e-04],
         [1.3680e-03, 7.1527e-02, 9.2711e-01]]),
 tensor([2, 1, 1, 2, 2, 1, 0, 2, 1, 2, 1, 0, 2, 0, 0, 0, 2, 2, 1, 0, 0, 2, 0, 2,
         2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 2, 1, 0, 2, 2, 1, 2, 1,
         0, 0, 2, 0, 0, 2, 1, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2,
         2, 1, 2, 0, 0, 2, 2, 2, 2, 1, 0, 1, 1, 2, 2, 0, 1, 1, 0, 0, 1, 2, 1, 2,
         0, 2, 1, 2, 2, 2, 0, 2, 0, 1, 2, 2, 2, 0, 2, 0, 1, 2, 2, 2, 1, 2, 2, 2,
         0, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 1, 2, 2, 2, 2, 1, 2, 0, 0, 2, 0, 0,
         2, 2, 0, 2, 2, 0, 2, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0,
         1, 0, 2, 1, 2, 0, 2, 0, 1, 1, 2, 0, 2, 2, 1, 2, 0, 1, 1, 0, 0, 2, 1, 0,
         2, 1, 1, 0, 0, 0, 0, 2])]
```

Our y categories are one of the 3 classes indicating the winner to be the Away team, Home team or a draw.

```python
data.y.classes

```




    ['A', 'D', 'H']


## The predictions as dataframe.  




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>home_team</th>
      <th>away_team</th>
      <th>real_winner</th>
      <th>predicted</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Everton</td>
      <td>Arsenal</td>
      <td>H</td>
      <td>H</td>
      <td>((Everton, 95.91294), (draw, 3.9653156), (Arse...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chelsea</td>
      <td>West Ham</td>
      <td>H</td>
      <td>H</td>
      <td>((Chelsea, 99.99604), (draw, 0.0032407162), (W...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Huddersfield</td>
      <td>Arsenal</td>
      <td>A</td>
      <td>D</td>
      <td>((Huddersfield, 0.6011091), (draw, 78.12512), ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brighton</td>
      <td>Burnley</td>
      <td>A</td>
      <td>A</td>
      <td>((Brighton, 3.833617e-06), (draw, 0.04196991),...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Liverpool</td>
      <td>Bournemouth</td>
      <td>H</td>
      <td>H</td>
      <td>((Liverpool, 100.0), (draw, 7.0475085e-07), (B...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Southampton</td>
      <td>Cardiff</td>
      <td>A</td>
      <td>A</td>
      <td>((Southampton, 0.0336317), (draw, 16.066362), ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fulham</td>
      <td>Man United</td>
      <td>A</td>
      <td>A</td>
      <td>((Fulham, 2.0389035e-07), (draw, 0.0077300384)...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Watford</td>
      <td>Everton</td>
      <td>H</td>
      <td>H</td>
      <td>((Watford, 66.78716), (draw, 32.949253), (Ever...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Crystal Palace</td>
      <td>West Ham</td>
      <td>D</td>
      <td>D</td>
      <td>((Crystal Palace, 2.0793211), (draw, 93.41345)...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cardiff</td>
      <td>West Ham</td>
      <td>H</td>
      <td>H</td>
      <td>((Cardiff, 99.99783), (draw, 0.0017247049), (W...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Leicester</td>
      <td>Fulham</td>
      <td>H</td>
      <td>H</td>
      <td>((Leicester, 99.9985), (draw, 0.00076819153), ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Southampton</td>
      <td>Tottenham</td>
      <td>H</td>
      <td>H</td>
      <td>((Southampton, 93.58026), (draw, 6.2125006), (...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Crystal Palace</td>
      <td>Brighton</td>
      <td>A</td>
      <td>A</td>
      <td>((Crystal Palace, 0.00065600086), (draw, 1.268...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Man City</td>
      <td>Watford</td>
      <td>H</td>
      <td>H</td>
      <td>((Man City, 99.991135), (draw, 0.0075131604), ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Huddersfield</td>
      <td>Bournemouth</td>
      <td>A</td>
      <td>A</td>
      <td>((Huddersfield, 0.0020392684), (draw, 4.043945...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Newcastle</td>
      <td>Everton</td>
      <td>H</td>
      <td>D</td>
      <td>((Newcastle, 45.162865), (draw, 53.98181), (Ev...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Man City</td>
      <td>Chelsea</td>
      <td>H</td>
      <td>H</td>
      <td>((Man City, 100.0), (draw, 2.8052786e-11), (Ch...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Tottenham</td>
      <td>Leicester</td>
      <td>H</td>
      <td>H</td>
      <td>((Tottenham, 99.995056), (draw, 0.003541528), ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Chelsea</td>
      <td>Wolves</td>
      <td>D</td>
      <td>D</td>
      <td>((Chelsea, 2.365934), (draw, 93.12464), (Wolve...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Arsenal</td>
      <td>Man United</td>
      <td>H</td>
      <td>H</td>
      <td>((Arsenal, 99.96575), (draw, 0.031816255), (Ma...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Liverpool</td>
      <td>Burnley</td>
      <td>H</td>
      <td>H</td>
      <td>((Liverpool, 99.98728), (draw, 0.010542426), (...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Wolves</td>
      <td>Newcastle</td>
      <td>D</td>
      <td>D</td>
      <td>((Wolves, 1.8315862), (draw, 91.23951), (Newca...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Cardiff</td>
      <td>Huddersfield</td>
      <td>D</td>
      <td>D</td>
      <td>((Cardiff, 14.202065), (draw, 83.98388), (Hudd...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Crystal Palace</td>
      <td>Watford</td>
      <td>A</td>
      <td>A</td>
      <td>((Crystal Palace, 0.021291165), (draw, 9.28893...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Chelsea</td>
      <td>Newcastle</td>
      <td>H</td>
      <td>H</td>
      <td>((Chelsea, 95.84944), (draw, 4.002416), (Newca...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>West Ham</td>
      <td>Arsenal</td>
      <td>H</td>
      <td>H</td>
      <td>((West Ham, 89.728134), (draw, 10.064338), (Ar...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Leicester</td>
      <td>Southampton</td>
      <td>A</td>
      <td>A</td>
      <td>((Leicester, 0.004756714), (draw, 5.1381216), ...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Brighton</td>
      <td>Liverpool</td>
      <td>A</td>
      <td>D</td>
      <td>((Brighton, 18.322472), (draw, 67.11339), (Liv...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Burnley</td>
      <td>Fulham</td>
      <td>H</td>
      <td>H</td>
      <td>((Burnley, 99.13705), (draw, 0.8168621), (Fulh...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Leicester</td>
      <td>Newcastle</td>
      <td>A</td>
      <td>A</td>
      <td>((Leicester, 0.0016775077), (draw, 2.3392487),...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Leicester</td>
      <td>Chelsea</td>
      <td>D</td>
      <td>D</td>
      <td>((Leicester, 0.95444053), (draw, 85.15614), (C...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Brighton</td>
      <td>Man City</td>
      <td>A</td>
      <td>A</td>
      <td>((Brighton, 0.1478293), (draw, 2.7528427), (Ma...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Burnley</td>
      <td>Arsenal</td>
      <td>A</td>
      <td>A</td>
      <td>((Burnley, 4.756038e-06), (draw, 0.03592212), ...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Liverpool</td>
      <td>Wolves</td>
      <td>H</td>
      <td>H</td>
      <td>((Liverpool, 99.969795), (draw, 0.016163945), ...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Crystal Palace</td>
      <td>Bournemouth</td>
      <td>H</td>
      <td>H</td>
      <td>((Crystal Palace, 99.99936), (draw, 0.00033819...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Fulham</td>
      <td>Newcastle</td>
      <td>A</td>
      <td>A</td>
      <td>((Fulham, 3.3353938e-11), (draw, 1.6640219e-05...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Man United</td>
      <td>Cardiff</td>
      <td>A</td>
      <td>A</td>
      <td>((Man United, 0.00021095584), (draw, 0.6491422...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Southampton</td>
      <td>Huddersfield</td>
      <td>D</td>
      <td>D</td>
      <td>((Southampton, 43.924297), (draw, 54.520206), ...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Watford</td>
      <td>West Ham</td>
      <td>A</td>
      <td>A</td>
      <td>((Watford, 1.571275e-08), (draw, 0.0007987406)...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Tottenham</td>
      <td>Everton</td>
      <td>D</td>
      <td>D</td>
      <td>((Tottenham, 12.236024), (draw, 82.49956), (Ev...</td>
    </tr>
  </tbody>
</table>
</div>




```python
learn.get_preds(ds_type=DatasetType.Valid)
```





### The learner. 


Our models is Linear model with 400 and 100 layers culminating in an output tensor of size 3. My output categories were A, D, H so that makes sense.

```python
learn

```




    Learner(data=TabularDataBunch;
    
    Train: LabelList (940 items)
    x: TabularList
    HomeTeam Sunderland; AwayTeam West Brom; HTR A; Referee S Attwell; LBH_na False; LBD_na False; LBA_na False; HS -1.1770; AS 1.1988; HF -1.0876; AF 0.5638; LBH -0.2038; LBD -0.6903; LBA -0.3624; FTHG -0.4246; FTAG -0.1694; HTHG -0.7922; HTAG 0.6260; HST -0.9981; AST 1.3226; HC 0.0907; AC 0.1299; HY -0.4927; AY 0.9943; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; ,HomeTeam West Ham; AwayTeam Middlesbrough; HTR D; Referee N Swarbrick; LBH_na False; LBD_na False; LBA_na False; HS 0.8697; AS -0.4445; HF 0.6956; AF 0.2811; LBH -0.3034; LBD -0.5185; LBA -0.3047; FTHG -0.4246; FTAG -0.1694; HTHG -0.7922; HTAG -0.7023; HST -0.9981; AST -0.3718; HC -0.5754; AC 0.1299; HY 0.3003; AY 0.9943; HR -0.2275; AR -0.2492; ht_goal_difference 0.0503; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; ,HomeTeam Swansea; AwayTeam Liverpool; HTR H; Referee M Oliver; LBH_na False; LBD_na False; LBA_na False; HS -1.0065; AS 1.4042; HF 0.1012; AF -0.5668; LBH 2.3357; LBD 0.6407; LBA -0.7804; FTHG -0.4246; FTAG 0.6731; HTHG 0.3993; HTAG -0.7023; HST -0.6346; AST 0.8990; HC -0.9085; AC 1.9573; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference -0.4110; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; ,HomeTeam Watford; AwayTeam Bournemouth; HTR A; Referee M Dean; LBH_na False; LBD_na False; LBA_na False; HS 0.5285; AS -0.0337; HF 1.8844; AF 0.2811; LBH -0.2138; LBD -0.6473; LBA -0.3624; FTHG 0.3299; FTAG 0.6731; HTHG -0.7922; HTAG 0.6260; HST 0.8194; AST -0.7954; HC -0.5754; AC 0.1299; HY 1.0933; AY 1.7719; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; ,HomeTeam Hull; AwayTeam Chelsea; HTR D; Referee A Taylor; LBH_na False; LBD_na False; LBA_na False; HS -1.0065; AS 2.2259; HF 0.6956; AF 1.1290; LBH 2.0867; LBD 0.3402; LBA -0.7666; FTHG -1.1792; FTAG 0.6731; HTHG -0.7922; HTAG -0.7023; HST -0.6346; AST 2.1698; HC -0.2424; AC 0.8608; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference 0.0503; ft_goal_difference -0.9316; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week -1.7115; Day -0.6029; Dayofyear -1.6613; Elapsed -2.1808; 
    y: CategoryList
    D,D,A,D,A
    Path: /home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data;
    
    Valid: LabelList (200 items)
    x: TabularList
    HomeTeam Watford; AwayTeam Brighton; HTR H; Referee J Moss; LBH_na True; LBD_na True; LBA_na True; HS 0.8697; AS -1.0607; HF -0.1960; AF 1.4117; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 0.3299; FTAG -1.0119; HTHG 0.3993; HTAG -0.7023; HST 0.0924; AST -1.6426; HC 0.7569; AC -0.9666; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference 1.1509; loss3_streak -0.2023; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.2023; home_loss5_streak -0.2825; home_last4pts 0.2870; away_last4pts -0.8329; win3_streak -0.1580; home_win3_streak -0.1580; Week 0.9939; Day -0.8237; Dayofyear 0.9797; Elapsed 1.3691; ,HomeTeam Wolves; AwayTeam Everton; HTR D; Referee C Pawson; LBH_na True; LBD_na True; LBA_na True; HS -0.4948; AS -1.0607; HF -0.7904; AF -1.1320; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 0.3299; FTAG 0.6731; HTHG 0.3993; HTAG 0.6260; HST -0.2711; AST 0.4754; HC -0.9085; AC 0.4954; HY -1.2857; AY -0.5609; HR -0.2275; AR 4.0092; ht_goal_difference 0.0503; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.2483; away_last4pts -0.2741; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.9939; Day -0.8237; Dayofyear 0.9797; Elapsed 1.3691; ,HomeTeam Arsenal; AwayTeam Wolves; HTR A; Referee S Attwell; LBH_na True; LBD_na True; LBA_na True; HS -0.6654; AS 0.1718; HF -0.4932; AF 1.4117; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG -0.4246; FTAG -0.1694; HTHG -0.7922; HTAG 0.6260; HST -0.6346; AST 0.4754; HC 1.7561; AC -0.9666; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference 0.1097; loss3_streak -0.6962; win5_streak 0.2768; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak 0.2768; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts 4.5688; away_last4pts -0.2741; win3_streak 0.5094; home_win3_streak 0.5094; Week 0.9939; Day -0.4926; Dayofyear 1.0059; Elapsed 1.3794; ,HomeTeam Man City; AwayTeam Man United; HTR H; Referee A Taylor; LBH_na True; LBD_na True; LBA_na True; HS 0.5285; AS -1.0607; HF 0.3984; AF 0.2811; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 1.0845; FTAG -0.1694; HTHG 0.3993; HTAG -0.7023; HST 0.0924; AST -1.2190; HC -0.2424; AC -1.3321; HY -0.4927; AY -0.5609; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference 1.1509; loss3_streak -0.6962; win5_streak 0.8374; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak 0.8374; home_loss3_streak -0.6962; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.2741; win3_streak 1.5104; home_win3_streak 1.5104; Week 0.9939; Day -0.4926; Dayofyear 1.0059; Elapsed 1.3794; ,HomeTeam Liverpool; AwayTeam Fulham; HTR H; Referee P Tierney; LBH_na True; LBD_na True; LBA_na True; HS 1.0402; AS -0.6499; HF 0.1012; AF -0.5668; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 0.3299; FTAG -1.0119; HTHG 0.3993; HTAG -0.7023; HST 0.8194; AST -0.3718; HC 0.0907; AC -0.6011; HY -0.4927; AY -0.5609; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference -0.9316; loss3_streak 1.2794; win5_streak -0.2839; loss5_streak 0.9943; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak 1.2794; home_loss5_streak 0.9943; home_last4pts -0.2483; away_last4pts -0.8329; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.9939; Day -0.4926; Dayofyear 1.0059; Elapsed 1.3794; 
    y: CategoryList
    H,D,D,H,H
    Path: /home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data;
    
    Test: LabelList (40 items)
    x: TabularList
    HomeTeam Everton; AwayTeam Arsenal; HTR H; Referee K Friend; LBH_na True; LBD_na True; LBA_na True; HS 1.5519; AS -0.8553; HF -0.7904; AF -0.5668; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG -0.4246; FTAG -1.0119; HTHG 0.3993; HTAG -0.7023; HST 0.4559; AST -0.7954; HC 1.0899; AC 0.4954; HY -0.4927; AY 1.7719; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference -0.4110; loss3_streak -0.2023; win5_streak 0.2768; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak 0.2768; home_loss3_streak -0.2023; home_loss5_streak -0.2825; home_last4pts -0.2483; away_last4pts -0.2741; win3_streak 1.1767; home_win3_streak 1.1767; Week -0.1128; Day -1.2652; Dayofyear -0.1309; Elapsed 2.1870; ,HomeTeam Chelsea; AwayTeam West Ham; HTR H; Referee C Kavanagh; LBH_na True; LBD_na True; LBA_na True; HS 0.3580; AS -0.4445; HF -0.7904; AF -1.1320; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 0.3299; FTAG -1.0119; HTHG 0.3993; HTAG -0.7023; HST 0.8194; AST -0.7954; HC 0.4238; AC -0.2356; HY 0.3003; AY -0.5609; HR -0.2275; AR -0.2492; ht_goal_difference -0.7936; ft_goal_difference -0.9316; loss3_streak -0.2023; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.2023; home_loss5_streak -0.2825; home_last4pts 0.2870; away_last4pts -0.2741; win3_streak -0.1580; home_win3_streak -0.1580; Week 0.1331; Day -1.2652; Dayofyear 0.1402; Elapsed 2.2935; ,HomeTeam Huddersfield; AwayTeam Arsenal; HTR A; Referee J Moss; LBH_na True; LBD_na True; LBA_na True; HS 0.1874; AS -0.4445; HF 1.8844; AF 0.2811; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG -0.4246; FTAG 0.6731; HTHG -0.7922; HTAG 1.9542; HST 0.4559; AST 0.0518; HC -0.2424; AC -1.6976; HY 1.0933; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference -1.6374; ft_goal_difference -0.4110; loss3_streak 5.7245; win5_streak -0.2839; loss5_streak 8.6552; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak 5.7245; home_loss5_streak 8.6552; home_last4pts -0.2483; away_last4pts -0.2741; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.4405; Day -1.4859; Dayofyear 0.3938; Elapsed 2.3932; ,HomeTeam Brighton; AwayTeam Burnley; HTR A; Referee S Attwell; LBH_na True; LBD_na True; LBA_na True; HS 0.3580; AS -0.4445; HF -0.7904; AF -1.1320; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG -0.4246; FTAG 1.5155; HTHG -0.7922; HTAG 0.6260; HST 0.4559; AST 0.4754; HC 1.0899; AC -0.6011; HY -0.4927; AY -0.5609; HR -0.2275; AR -0.2492; ht_goal_difference 0.8941; ft_goal_difference 1.1509; loss3_streak 2.2672; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak 2.2672; home_loss5_streak -0.2825; home_last4pts 0.8222; away_last4pts -0.2741; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.4405; Day -1.4859; Dayofyear 0.3938; Elapsed 2.3932; ,HomeTeam Liverpool; AwayTeam Bournemouth; HTR H; Referee A Taylor; LBH_na True; LBD_na True; LBA_na True; HS 1.0402; AS 0.1718; HF 0.9928; AF -1.4147; LBH -0.3034; LBD -0.3468; LBA -0.3278; FTHG 1.0845; FTAG -1.0119; HTHG 1.5908; HTAG -0.7023; HST 1.5465; AST -0.7954; HC 0.7569; AC 0.1299; HY 0.3003; AY 0.2167; HR -0.2275; AR -0.2492; ht_goal_difference -1.6374; ft_goal_difference -1.4522; loss3_streak -0.2023; win5_streak -0.2839; loss5_streak -0.2825; away_win3_streak 0.0000; away_win5_streak 0.0000; away_loss3_streak 0.0000; away_loss5_streak 0.0000; home_win5_streak -0.2839; home_loss3_streak -0.2023; home_loss5_streak -0.2825; home_last4pts -0.7835; away_last4pts -0.2741; win3_streak -0.4916; home_win3_streak -0.4916; Week 0.4405; Day -1.4859; Dayofyear 0.3938; Elapsed 2.3932; 
    y: EmptyLabelList
    ,,,,
    Path: /home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data, model=TabularModel(
      (embeds): ModuleList(
        (0): Embedding(27, 10)
        (1): Embedding(27, 10)
        (2): Embedding(4, 3)
        (3): Embedding(23, 9)
        (4): Embedding(3, 3)
        (5): Embedding(3, 3)
        (6): Embedding(3, 3)
      )
      (emb_drop): Dropout(p=0.0, inplace=False)
      (bn_cont): BatchNorm1d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (layers): Sequential(
        (0): Linear(in_features=80, out_features=400, bias=True)
        (1): ReLU(inplace=True)
        (2): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Linear(in_features=400, out_features=100, bias=True)
        (4): ReLU(inplace=True)
        (5): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): Linear(in_features=100, out_features=3, bias=True)
      )
    ), opt_func=functools.partial(<class 'torch.optim.adam.Adam'>, betas=(0.9, 0.99)), loss_func=FlattenedLoss of CrossEntropyLoss(), metrics=[<function accuracy at 0x7fc508c12710>], true_wd=True, bn_wd=True, wd=0.01, train_bn=True, path=PosixPath('/home/sidravic/Dropbox/code/workspace/football-data/notebooks/footy/EPL_Predictions/data'), model_dir='models', callback_fns=[functools.partial(<class 'fastai.basic_train.Recorder'>, add_time=True, silent=False)], callbacks=[], layer_groups=[Sequential(
      (0): Embedding(27, 10)
      (1): Embedding(27, 10)
      (2): Embedding(4, 3)
      (3): Embedding(23, 9)
      (4): Embedding(3, 3)
      (5): Embedding(3, 3)
      (6): Embedding(3, 3)
      (7): Dropout(p=0.0, inplace=False)
      (8): BatchNorm1d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (9): Linear(in_features=80, out_features=400, bias=True)
      (10): ReLU(inplace=True)
      (11): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (12): Linear(in_features=400, out_features=100, bias=True)
      (13): ReLU(inplace=True)
      (14): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (15): Linear(in_features=100, out_features=3, bias=True)
    )], add_time=True, silent=False)



