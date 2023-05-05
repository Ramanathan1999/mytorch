from pathlib import Path
import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """
        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        
        pred=[]
        for i in range(len(y_probs[1])):
            path_prob *= np.max(y_probs[:,i,:])
            index= np.argmax(y_probs[:,i,:])
            pred.append(index)
            
        if pred[0]!=blank:
            decoded_path.append(self.symbol_set[pred[0]-1])
            
        for x in range(1,len(pred)):
            if pred[x]!=blank and pred[x]!= pred[x-1]:
                decoded_path.append(self.symbol_set[pred[x]-1])

        decoded_path = ''.join(decoded_path)                
        return decoded_path, path_prob

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def InitializePaths(self, SymbolSet, y):
        InitialBlankPathScore,InitialPathScore = {},{}
        path = ''
        InitialBlankPathScore[path] = y[0]
        for i,c in enumerate(SymbolSet): 
            InitialPathScore[c] = y[i+1]
        return InitialBlankPathScore, InitialPathScore

    def Prune(self, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore, PrunedPathScore, scorelist  = {}, {},[]
        for p in BlankPathScore.keys():
            scorelist.append(BlankPathScore[p])
        for p in PathScore.keys():
            scorelist.append(PathScore[p])

        scorelist.sort(reverse=True)

        if BeamWidth < len(scorelist):
            cutoff = scorelist[BeamWidth]
        else:
            cutoff = scorelist[-1]

        for p in BlankPathScore.keys():
            if BlankPathScore[p] > cutoff:
                PrunedBlankPathScore[p] = BlankPathScore[p]
        for p in PathScore.keys():
            if PathScore[p] > cutoff:
                PrunedPathScore[p] = PathScore[p]

        return PrunedBlankPathScore, PrunedPathScore

    def Extend(self, SymbolSet, y, BlankPathScore, PathScore):
        UpdatedBlankPathScore, UpdatedPathScore = {} ,{}
        
        for path in BlankPathScore.keys():
            UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]
        for path in PathScore.keys():
            if path in UpdatedBlankPathScore.keys():
                UpdatedBlankPathScore[path] += PathScore[path]*y[0]
            else:
                UpdatedBlankPathScore[path] = PathScore[path]*y[0]

        for path in BlankPathScore.keys():
            for i,c in enumerate(SymbolSet):
                newpath = path + c
                UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]
        for path in PathScore.keys():
            for i,c in enumerate(SymbolSet):
                newpath = path if (c == path[-1]) else path + c 
                if newpath in UpdatedPathScore.keys():
                    UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
                else: 
                    UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
        
        return  UpdatedBlankPathScore,UpdatedPathScore

    def Merge(self, BlankPathScore, PathScore):
        Finalscores = {}
        for path in BlankPathScore.keys():
            Finalscores[path] = BlankPathScore[path]
        for path in PathScore.keys():
            if path in Finalscores.keys():
                Finalscores[path]+= PathScore[path]
            else:
                Finalscores[path] = PathScore[path]

        max_score = sorted(Finalscores.values(),reverse=True)
        for i in Finalscores.keys():
            if Finalscores[i] == max_score[0]:
                best_path = i 

        return best_path, Finalscores

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        T = y_probs.shape[1]
        SymbolSet = self.symbol_set
        BeamWidth = self.beam_width

        NewBlankPathScore, NewPathScore = self.InitializePaths(SymbolSet, y_probs[:,0, :])
        for t in range(1, T):
            BlankPathScore, PathScore = self.Prune(NewBlankPathScore, NewPathScore, BeamWidth)
            NewBlankPathScore, NewPathScore = self.Extend(SymbolSet, y_probs[:,t,:], BlankPathScore, PathScore)
        bestPath, FinalPathScore = self.Merge(NewBlankPathScore, NewPathScore) 
        return bestPath, FinalPathScore
    





























