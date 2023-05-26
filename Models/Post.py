import math


class Post:
    def __init__(self, discussion_id, post_id, message, parent_id, author_id, date):
        self.discussion_id = discussion_id
        self.post_id = post_id
        self.message = message
        self.parent_id = parent_id
        self.thread = []
        self.username = author_id
        self.date = date

    def set_thread(self, thread):
        self.thread = thread

    def update_message(self, message):
        self.message = message
