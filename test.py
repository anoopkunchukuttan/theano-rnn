import numpy as np
import theano
import theano.tensor as T

class OutputLayer:     
    def __init__(self, n_in, n_out, activation, name): 
    
        self.n_in=n_in
        self.n_out=n_out

        W_init= np.asarray(np.random.uniform(size=(n_in, n_out),
                                                  low=-.01, high=.01),
                                                  dtype=theano.config.floatX) 
        self.W_out = theano.shared(value=W_init, name='W_out_{}'.format(name))

        by_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init, name='by_{}'.format(name))

        self.params=[self.W_out,self.by]

    def process_input(self, indata): 
        return self.activation(T.dot(indata, self.W_out) + self.by)

    def get_params(self): 
        return self.params

class HiddenLayer: 

    def __init__(self, n_in, n_out, activation, name): 
        self.n_in=n_in
        self.n_out=n_out

        W_init= np.asarray(np.random.uniform(size=(n_in, n_out),
                                                  low=-.01, high=.01),
                                                  dtype=theano.config.floatX) 
        self.W_hid = theano.shared(value=W_init, name='W_hid_{}'.format(name))

        b_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=b_init, name='bh_{}'.format(name))

        W_init= np.asarray(np.random.uniform(size=(n_out, n_out),
                                                  low=-.01, high=.01),
                                                  dtype=theano.config.floatX) 
        self.W_rec = theano.shared(value=W_init, name='W_rec_{}'.format(name))

        hrec_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.hrec0 = theano.shared(value=hrec_init, name='hrec_{}'.format(name))

        #TODO: hrec0 also a paramter to be learnt
        self.params=[self.W_rec,self.W_hid,self.bh]

    def process_input(self, indata, h_tm1): 
        h_t = self.activation(T.dot(indata, self.W_hid) + \
                                  T.dot(h_tm1, self.W_rec) + self.bh)
        return h_t

    def get_params(self): 
        return self.params

    def get_init_state(self): 
        return hrec0 

class RNNMultiOut:         
    def __init__(self, n_in, n_hid, n_out, n_groups): 
        self.n_in=n_in
        self.n_hid=n_hid
        self.n_out=n_out

        self.h_l=HiddenLayer(n_in, n_hid, T.nnet.sigmoid, 'h1')
        self.o_ls=[ OutputLayer(n_hid, n_out, T.nnet.softmax, 'o{}'.format(i) for i in xrange(n_groups) ]

        x=T.matrix()
        y=T.matrix()
        g=T.vector()

        #### function for processing a time step 
        # recurrent function (using tanh activation function) and linear output
        # activation function
        def step(x_t, h_tm1):
            h_t = self.h_l.process_input(x_t, h_tm1)
            y_ts= [ self.o_ls[i].process_input(h_t) for i in xrange(n_groups) ]

            return h_t, y_t[0], y_t[1]
                        
        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        [self.h, self.y0_pred, self.y1_pred], _ = theano.scan(step,
                                               sequences=self.x,
                                               outputs_info=[self.h_l.get_init_state(), None, None])


    def load_data(data_fname):
        pass  

def main(): 
    loss1=

if __name__ == '__main__': 
    main()
