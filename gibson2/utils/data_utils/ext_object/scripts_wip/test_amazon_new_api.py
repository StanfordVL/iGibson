from IPython import embed
from aws_requests_auth.aws_auth import AWSRequestsAuth
import boto3
import requests
import argparse
import numpy as np


def main(args):
    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': 'Atzr|IwEBILncU1UpR8HLAvBHk_ONWyqW2jTyYf2gP7iQWlja_XG1VjvUi8RHZfsNAI87XFU-UQRqgrw5wPH9ekcnxp5O5jeN4HGU_5Xp3RCU29hKUBEtusGmrAyMlqG4C_kZihaacYStbyzYrd_MZMbp3kLZIRz8Ku0EL-Edup-kmpJ8JzB_sJllBK1_48QQEhm7Z6yxM13cnr-eYBkValoV8sYKnLIolkVRE01Kt_PXh-rJplB2-QgQBRMp5-mMq1Z0KIjGIqOoid75Wh8GvGUhirmAOXFF1sq04K2RK-3BRQcYgAONp2Zw8tjxKZR6Uy3_-4vIPa8',
        'client_id': 'amzn1.application-oa2-client.2b1d75a51a2f4d00a0e4f7ba9c772d0b',
        'client_secret': 'dba0959db75fe031de5da70a0bc3e1ffb43349a8aa31d14fa097f4c96b9a2b59',
    }

    url = 'https://api.amazon.com/auth/o2/token'
    r = requests.post(url, data=payload)
    assert r.status_code == 200
    json_result = r.json()
    assert 'access_token' in json_result
    access_token = json_result['access_token']

    # url = 'https://sellingpartnerapi-na.amazon.com/catalog/v0/items'
    # payload = {
    #     'MarketplaceId': market_id
    # }
    # headers = {
    #     'host': 'https://sellingpartnerapi-na.amazon.com',
    #     'x-amz-access-token': access_token,
    #     'user-agent': 'my-app/0.0.1',
    #     'x-amz-date': '20190430T123600Z',
    # }
    # r = requests.get(url, params=payload)
    # print(r.status_code)
    # print(r.text)

    AMZ_Client = boto3.client('sts',
                              aws_access_key_id='AKIATVVECHMWB7T4JHO4',
                              aws_secret_access_key='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',
                              region_name='us-east-1')
    res = AMZ_Client.assume_role(
        RoleArn='arn:aws:iam::252673407788:role/SellingPartnerAPI',
        RoleSessionName='SellingPartnerAPI'
    )
    Credentials = res["Credentials"]
    AccessKeyId = Credentials["AccessKeyId"]
    SecretAccessKey = Credentials["SecretAccessKey"]
    SessionToken = Credentials["SessionToken"]

    endpoint = 'https://sellingpartnerapi-na.amazon.com/catalog/v0/items'
    market_id = 'ATVPDKIKX0DER'

    auth = AWSRequestsAuth(aws_access_key=AccessKeyId,
                           aws_secret_access_key=SecretAccessKey,
                           aws_token=SessionToken,
                           aws_host='sellingpartnerapi-na.amazon.com',
                           aws_region='us-east-1',
                           aws_service='execute-api')
    headers = {'x-amz-access-token': access_token}

    for query in args.obj_list:
        payload = {
            'MarketplaceId': market_id,
            'Query': query,
        }

        r = requests.get(endpoint, auth=auth, headers=headers, params=payload)
        obj = r.json()
        dims = []
        for item in obj['payload']['Items']:
            attribute_sets = item['AttributeSets'][0]
            if 'ItemDimensions' in attribute_sets:
                item_dim = attribute_sets['ItemDimensions']
                has_valid_dim = 'Width' in item_dim and 'Length' in item_dim \
                    and 'Height' in item_dim and 'Weight' in item_dim
                # no valid dimension, skip
                if not has_valid_dim:
                    continue

                item_dim_array = [
                    item_dim['Width']['value'],
                    item_dim['Length']['value'],
                    item_dim['Height']['value'],
                    item_dim['Weight']['value'],
                ]
                dims.append([float(item) for item in item_dim_array])
        print(query)
        print(len(dims))
        for dim in dims:
            print(dim)
        print(np.median(dims, axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_list', nargs='+', required=True, type=str)
    args = parser.parse_args()
    main(args)
