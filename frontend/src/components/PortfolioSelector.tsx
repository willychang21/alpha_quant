import React, { useState, useRef, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import { ChevronDown, Plus, Pencil, Trash2, Check, X } from 'lucide-react';

interface Portfolio {
  id: string;
  name: string;
}

interface PortfolioSelectorProps {
  currentPortfolioId: string | null;
  portfolios: Portfolio[];
  onSelect: (id: string) => void;
  onCreate: (name: string) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
}

const PortfolioSelector: React.FC<PortfolioSelectorProps> = ({ 
  currentPortfolioId, 
  onSelect, 
  portfolios, 
  onCreate, 
  onRename, 
  onDelete 
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null); // Track which portfolio is being edited
  const [editingName, setEditingName] = useState(''); // Track the editing value
  const editInputRef = useRef<HTMLInputElement>(null);

  const portfolioList = Array.isArray(portfolios) ? portfolios : [];
  const currentPortfolio = portfolioList.find(p => p.id === currentPortfolioId) || portfolioList[0];
  
  // Focus input when editing starts
  useEffect(() => {
    if (editingId !== null && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingId]);
  
  const handleCreate = () => {
    if (newName.trim()) {
      onCreate(newName);
      setNewName('');
      setIsCreating(false);
    }
  };

  const handleStartEdit = (portfolio: Portfolio, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(portfolio.id);
    setEditingName(portfolio.name);
  };

  const handleSaveEdit = () => {
    if (editingName.trim() && editingId !== null) {
      onRename(editingId, editingName.trim());
      setEditingId(null);
      setEditingName('');
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditingName('');
  };

  const handleEditKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveEdit();
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };
  
  const handleSelect = (id: string) => {
    if (editingId !== null) return; // Don't select when editing
    onSelect(id);
    setIsMenuOpen(false);
  };

  const handleOpenCreate = () => {
    setIsMenuOpen(false);
    setNewName('');
    setIsCreating(true);
  };
  
  const handleDelete = (portfolio: Portfolio, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Delete "${portfolio.name}"? This cannot be undone.`)) {
      onDelete(portfolio.id);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <Popover open={isMenuOpen} onOpenChange={setIsMenuOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            className="w-[240px] h-12 rounded-full border-none bg-secondary/50 hover:bg-secondary/80 text-foreground px-5 py-2 flex items-center justify-between transition-all shadow-sm hover:shadow-md"
          >
            <span className="font-medium text-sm truncate mr-3">
                {currentPortfolio?.name || 'Select Portfolio'}
            </span>
            <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[260px] p-0 bg-popover border border-border shadow-xl rounded-2xl overflow-hidden" align="start">
          <Command className="bg-transparent">
            <div className="p-2 border-b border-border/40">
              <CommandInput 
                placeholder="Search..." 
                className="h-9 rounded-full bg-secondary px-4 text-sm outline-none focus:bg-secondary/80 transition-colors placeholder:text-muted-foreground/70"
              />
            </div>
            <CommandList className="p-2 max-h-[300px] overflow-y-auto custom-scrollbar">
              <CommandEmpty>No portfolio found.</CommandEmpty>
              <CommandGroup heading="Portfolios" className="px-0 py-1">
                {portfolioList.map((p) => (
                  <CommandItem
                    key={p.id}
                    value={p.name}
                    onSelect={() => handleSelect(p.id)}
                    className="justify-between py-3 px-4 mb-1 rounded-full transition-all duration-200 group cursor-pointer hover:bg-secondary aria-selected:bg-primary/10 aria-selected:text-primary"
                  >
                    {editingId === p.id ? (
                      // Editing mode
                      <div className="flex items-center gap-2 flex-1" onClick={(e) => e.stopPropagation()}>
                        <Input
                          ref={editInputRef}
                          value={editingName}
                          onChange={(e) => setEditingName(e.target.value)}
                          onKeyDown={handleEditKeyDown}
                          className="h-8 text-sm flex-1 bg-white dark:bg-black/20 border-none shadow-none focus-visible:ring-0"
                          onClick={(e) => e.stopPropagation()}
                        />
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-8 w-8 rounded-full text-success hover:bg-success/10"
                          onClick={handleSaveEdit}
                        >
                          <Check className="h-4 w-4" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-8 w-8 rounded-full text-muted-foreground hover:bg-muted"
                          onClick={handleCancelEdit}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    ) : (
                      // Display mode
                      <>
                        <div className="flex flex-col gap-0.5 overflow-hidden flex-1">
                          <p className="font-medium text-sm truncate group-aria-selected:text-primary">{p.name}</p>
                          {p.id === currentPortfolioId && (
                            <span className="text-[10px] font-bold text-primary uppercase tracking-wider">Active</span>
                          )}
                        </div>
                        
                        <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 rounded-full opacity-0 group-hover:opacity-100 hover:bg-primary/10 hover:text-primary transition-opacity"
                            onClick={(e) => handleStartEdit(p, e)}
                            title="Rename"
                          >
                            <Pencil className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 rounded-full opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-opacity"
                            onClick={(e) => handleDelete(p, e)}
                            disabled={portfolioList.length <= 1}
                            title="Delete"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </>
                    )}
                  </CommandItem>
                ))}
              </CommandGroup>
              <CommandSeparator className="bg-border/40 my-2 mx-2" />
              <CommandGroup className="px-0 py-1">
                <CommandItem 
                  onSelect={handleOpenCreate}
                  className="py-3 px-4 rounded-full transition-all duration-200 cursor-pointer text-sm font-medium text-muted-foreground justify-center hover:bg-primary/5 hover:text-primary aria-selected:bg-primary/10 aria-selected:text-primary"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Create New Portfolio
                </CommandItem>
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      <Dialog open={isCreating} onOpenChange={setIsCreating}>
        <DialogContent className="material-card p-6 sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle className="text-2xl font-normal">Create New Portfolio</DialogTitle>
          </DialogHeader>
          <div className="py-6">
            <Input 
              placeholder="Portfolio Name" 
              value={newName} 
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
              className="input-material text-lg"
              autoFocus
            />
          </div>
          <DialogFooter className="gap-2">
            <Button variant="ghost" onClick={() => setIsCreating(false)} className="rounded-full">Cancel</Button>
            <Button 
              onClick={handleCreate}
              className="btn-material bg-primary text-primary-foreground px-6"
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default PortfolioSelector;
