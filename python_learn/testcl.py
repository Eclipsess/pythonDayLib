class a(object):
    x = 1
    def __init__(self):
        print 'create a'
        self.y = self.x + 1
        print self.y
    def xx(self):
        print self.x

aa = a()
aa.x = 2
print a.x
