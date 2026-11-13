class Post:
    def publish(self, when):
        return when


def go_live(post: Post):
    post.publish("now")
