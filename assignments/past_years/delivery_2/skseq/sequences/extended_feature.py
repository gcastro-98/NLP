from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures

# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)
            
            
        '''Adding new features'''
        
        #Initial letter upper
        if word[0].isupper():
            # Generate feature name.
            feat_name = "start_upper::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
                
        #All words upper
        if word.isupper():
            # Generate feature name.
            feat_name = "upper::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
                
        #4 digits (for years)
        if (word.isdigit() & (len(word) == 4)):
            # Generate feature name.
            feat_name = "year::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
        
        
        #Finishes in -day
        if (word[-3:] == 'day'):
            # Generate feature name.
            feat_name = "day::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
                
        #Contains dots in the middle of the word
        if '.' in word:
            # Generate feature name.
            feat_name = "dot::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
                
        #Finishes in -ese / -an (italian, american, japanese...)
        if ((word[-3:] == 'ese') | (word[-2:] == 'an') | (word[-3:] == 'ans')):
            # Generate feature name.
            feat_name = "esean::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
                
        # Contains ' in the word
        if "'" in word:
            # Generate feature name.
            feat_name = "'::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
                
        #Has - in the word
        if '-' in word:
            # Generate feature name.
            feat_name = "-::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
                
        #Previous word features
        
        if pos != 0:
            x_prev = sequence.x[pos-1]
            # Get word name from ID.
            if isinstance(x_prev, str):
                x_name = x_prev
            else:
                x_name = self.dataset.x_dict.get_label_name(x_prev)
            prev_word = str(x_name)
            
            #If previous word through
            if prev_word == 'through':
                # Generate feature name.
                feat_name = "through::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
               
            #If previous word in
            if prev_word == 'in':
                # Generate feature name.
                feat_name = "in::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word at
            if prev_word == 'at':
                # Generate feature name.
                feat_name = "at::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
            
            #If previous word from
            if prev_word == 'from':
                # Generate feature name.
                feat_name = "from::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
            
            #If previous word before
            if prev_word == 'before':
                # Generate feature name.
                feat_name = "before::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
            
            #If previous word of
            if prev_word == 'of':
                # Generate feature name.
                feat_name = "of::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
            
            #If previous word the
            if prev_word == 'the':
                # Generate feature name.
                feat_name = "the::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
            
            #If previous word with
            if prev_word == 'with':
                # Generate feature name.
                feat_name = "with::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word which
            if prev_word == 'which':
                # Generate feature name.
                feat_name = "which::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word when
            if prev_word == 'when':
                # Generate feature name.
                feat_name = "when::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word after
            if prev_word == 'after':
                # Generate feature name.
                feat_name = "after::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word early
            if prev_word == 'early':
                # Generate feature name.
                feat_name = "early::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word late
            if prev_word == 'late':
                # Generate feature name.
                feat_name = "late::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word on
            if prev_word == 'on':
                # Generate feature name.
                feat_name = "on::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word until
            if prev_word == 'until':
                # Generate feature name.
                feat_name = "until::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word by
            if prev_word == 'by':
                # Generate feature name.
                feat_name = "by::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
            #If previous word finishes in ern
            if prev_word[-3:] == 'ern':
                # Generate feature name.
                feat_name = "ern::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
        
        
        

        return features
