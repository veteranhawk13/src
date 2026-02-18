class BrowserHistory {
      struct node{
        string history;
        node *next;
        node *prev;
    };
    node *head;
    node *curr;
public:
    BrowserHistory(string homepage) {
        head=new node;
        head->history=homepage;
        head->next=NULL;
        head->prev=NULL;
        curr=head;
    }
    void visit(string url) {
        node *newNode=new node;
        newNode->history=url;
        newNode->next=NULL;
        node *temp=curr->next;
        while(temp!=NULL){
            node *del=temp;
            temp=temp->next;
            delete del;
        }
        curr->next=NULL;
        curr->next=newNode;
        newNode->prev=curr;
        curr=newNode;
    }
    string back(int steps) {
        for(int i=1;i<=steps && curr->prev != NULL;i++){
            curr=curr->prev;
        }
        return curr->history;
    }
    
    string forward(int steps) {
        for(int i=1;i<=steps && curr->next != NULL ;i++){
            curr=curr->next;
        }
        return curr->history;
    }
};

/**
 * Your BrowserHistory object will be instantiated and called as such:
 * BrowserHistory* obj = new BrowserHistory(homepage);
 * obj->visit(url);
 * string param_2 = obj->back(steps);
 * string param_3 = obj->forward(steps);
 */
