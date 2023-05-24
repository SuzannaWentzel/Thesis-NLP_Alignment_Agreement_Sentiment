import stanza
import string


"""
Recursive function to get the rules from a constituency parse
"""
def get_rules_string(tree):
    rule_string = tree.label + " ->"

    rules = []
    has_right_hand_side = False

    for child in tree.children:
        if len(child.children) == 0:
            continue
        has_right_hand_side = True
        rule_string += " " + child.label
        rules += (get_rules_string(child))

    if has_right_hand_side:
        rules.insert(0, rule_string)

    return rules


def preprocess_message_syntactic(message):
    # To lowercase
    message = message.lower()

    # Get sentences
    nlp_sentences = stanza.Pipeline(lang='en', processors='tokenize')
    doc_sentences = nlp_sentences(message)
    rules = []
    for sentence in doc_sentences.sentences:
        # Remove punctuation
        sentence_text = sentence.text
        sentence_text = sentence_text.translate(str.maketrans('', '', string.punctuation))
        # Get constituency
        nlp_constituency = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        doc_constituency = nlp_constituency(sentence_text)
        for sentence_constituency in doc_constituency.sentences:
            tree = sentence_constituency.constituency
            children = tree.children
            # TODO: extract rules from the tree.
            print(children)
            print(type(children))
            print(children[0])
            print(type(children[0]))
            print(children[0].label)
            # print(len(children))
            print("nf", get_rules_string(children[0]))


"""
Preprocess message for syntactical alignment measure
"""
def get_preprocessed_messages_for_syntactic(discussions):
    print('[TASK] preprocessing messages')
    preprocessed_posts = {}
    # get all the preprocessed posts
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            preprocessed = preprocess_message_syntactic(post.message)
            preprocessed_posts[str(i) + '-' + str(j)] = preprocessed
    print('[INFO] task completed')
    return preprocessed_posts


"""
Computes the syntactical alignment between messages
"""
def get_syntactical_alignment(discussions, preprocessed_messages):
    return True

# preprocessed_message = preprocess_message_syntactic('Hi. I was wondering, how are you? I do not agree.')
# preprocessed_message = preprocess_message_syntactic('I am a teacher.')
