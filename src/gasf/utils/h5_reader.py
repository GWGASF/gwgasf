import h5py

class h5_thang():
    
    """
    This funtion severs for loading all groups/name of h5 compressed data to a dictionary.
    The data can be eazily acess by the proivded groups/name. 
    """
    
    def __init__(
        self, 
        file
    ):

        self.file = file
        
    def h5_keys(
        self, 
        verbose=False
    )->list: 
        
        """
        Args:
            verbose (bool, optional): Defaults to False. If true print out all groups/name.
        Returns:
            list: A list of groups/name.
        """
        
        items = []
        def func(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append(name)

        f = h5py.File(self.file, 'r', locking=False)
        f.visititems(func)

        if verbose:
            for item in items:
                print(item)
        return items
    
    def h5_data(
        self,
        items: list=None,
        verbose: bool=False 
    )->dict:

        """
        Args:
            items (list): A list of groups/name to require from the compressed h5 file.
        Returns:
            dict: A dictonary that contains all direct acessable data
        """
        if items == None:
            items = self.h5_keys(verbose=False)
            
        if verbose:
            for item in items:
                print(item)
                
        data_dict = {}
        with h5py.File(self.file , 'r', locking=False) as h1:
            for item in items:
                data_dict[item] = h1[item][()]
                    
        return data_dict