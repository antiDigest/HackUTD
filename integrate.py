from bayesnet import BayesNet

from wit import Wit
import pandas as pd

access_token = 'YX2I7WOEAJXJ7QGSAFOOSFTUF3AXFGXU'


def my_action(request):
    print('Received from user...', request['text'])


def send(request, response):
    print('Sending to user...', response['text'])

actions = {
    'send': send,
    'my_action': my_action,
}

client = Wit(access_token=access_token, actions=actions)

bayes = BayesNet()
docBot = bayes.prob_symptoms

session_id = 'my-user-session-42'
context0 = {}
context1 = client.run_actions(
    session_id, 'I have nausea and shortness of breath', context0)
# print('The session state is now: ' + str(context1))

resp = client.converse('my-user-session-42',
                       'I have nausea and shortness of breath', {})

# print(str(resp))
lst = []

lst.append(dict(resp)['entities']['Symptoms'][0]['value'])
lst.append(dict(resp)['entities']['Symptoms'][1]['value'])

sol_dict = docBot(lst)

salt = sorted(sol_dict['_'.join(lst)].items(),
              key=lambda x: x[1], reverse=True)
# print salt

if salt[0][1] - salt[1][1] < 0.35:
    stmt = "You have a", salt[0][1] * \
        100, "% probability of suffering from", salt[0][0]
else:
    stmt = "You are suffering from either", salt[0][0], "or", salt[1][0]

print(stmt)
# Response to User -- stuck on this
response = stmt
